use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use realfft::FftNum;
use rtrb::{Consumer, Producer, RingBuffer};
use rtsan_standalone::{blocking, nonblocking};

use crate::utilities::{compute_tail_block_size, next_power_of_2};
use crate::{FFTConvolver, FFTConvolverError};

/// Capacity of each handoff ring, in tail blocks.
///
/// Only two blocks can be in flight in normal operation (see [`Link`]), so this
/// leaves headroom before the recovery path in
/// [`exchange_tail_block`](ThreadedFFTConvolver::exchange_tail_block) triggers.
const RING_BLOCKS: usize = 4;

/// Upper bound on how long the worker sleeps before looking for work itself.
/// See [`Semaphore::wait`]; on the paths this crate controls it never expires.
const WAKE_TIMEOUT: Duration = Duration::from_millis(100);

/// A binary semaphore built on the standard library's thread parking.
///
/// [`post`](Self::post) is called from the audio thread and must not block. The
/// standard library implements it with the primitive that is appropriate for
/// waking a thread on each platform: a futex wake on Linux, a
/// `dispatch_semaphore_signal` on macOS, and `WakeByAddressSingle`/keyed events
/// on Windows. None of them takes a lock, which rules out the priority
/// inversion a `Mutex` + `Condvar` pair would introduce.
///
/// The token is capped at one, which is all this design needs: the worker
/// drains every block that is available before waiting again, so a second post
/// arriving while the worker is busy has nothing to add. Posting while the
/// worker is running is never lost either, the token simply makes the next
/// [`wait`](Self::wait) return immediately.
///
/// The thread to wake is registered by the worker rather than fixed when the
/// convolver is built, because with [`ThreadedFFTConvolver::split`] the worker
/// runs on a thread this crate never sees. Reading the slot is a plain atomic
/// load, so the audio thread stays wait-free whether or not it has been set.
#[derive(Debug, Default)]
struct Semaphore {
    target: OnceLock<Thread>,
}

impl Semaphore {
    /// Registers the thread that [`post`](Self::post) wakes. The first
    /// registration wins; see [`TailWorker::set_waker`].
    fn register(&self, thread: Thread) {
        let _ = self.target.set(thread);
    }

    /// Wakes the worker. Called from the audio thread, once per tail block.
    ///
    /// Does nothing while no thread is registered, which happens when the
    /// worker has not started yet or when the caller drives it by polling
    /// [`TailWorker::run_pending`]. No work is lost either way: the worker
    /// drains everything that is ready before it waits again.
    ///
    /// RealtimeSanitizer flags the wake as an unsafe library call, because the
    /// syscall behind it is not on its list of operations with a bounded cost.
    /// It is suppressed here rather than by dropping the annotation from
    /// [`process`](ThreadedFFTConvolver::process), so everything else on the
    /// audio path stays checked. The suppression is sound for this one call:
    /// waking a thread does not take a lock, allocate, or wait for the woken
    /// thread, and it runs once per tail block, not once per callback.
    #[inline]
    fn post(&self) {
        if let Some(thread) = self.target.get() {
            rtsan_standalone::scoped_disabler! {
                thread.unpark();
            }
        }
    }

    /// Blocks until a token is available, or until the safety net expires.
    /// Only ever called by the worker.
    ///
    /// The timeout covers the window in which there is no thread to wake yet:
    /// between [`ThreadedFFTConvolver::split`] and the caller's first
    /// [`TailWorker::run`], a post has nowhere to go, and a worker that parked
    /// in the meantime would otherwise have to be woken by a post that already
    /// happened. Nothing is lost, because the worker re-reads its queue and the
    /// shutdown flag on every pass; the timeout only bounds how long that takes.
    /// Wakes that do arrive are immediate, so this never delays real work.
    #[inline]
    fn wait() {
        thread::park_timeout(WAKE_TIMEOUT);
    }
}

/// State shared between the audio thread and the worker.
#[derive(Debug, Default)]
struct Shared {
    /// Tells the worker to leave its loop.
    shutdown: AtomicBool,
    /// Index of the first block that has to be convolved with a cleared tail
    /// state. Written by the audio thread, applied by the worker once it gets
    /// to that block, which keeps the reset in sync with the sample stream
    /// without the audio thread ever waiting.
    resync_at: AtomicU64,
    /// Number of blocks whose results the worker has published.
    completed: AtomicU64,
    semaphore: Semaphore,
}

/// The tail stage of a [`ThreadedFFTConvolver`], to be run on a thread you own
///
/// Returned by [`ThreadedFFTConvolver::split`] for callers who would rather
/// place the work themselves than have a thread spawned for them. Two reasons
/// to want that: setting up the thread (real-time priority, affinity, a name)
/// without a callback, and running the tails of several convolvers on one
/// thread instead of one thread per instance.
///
/// The simple case is [`run`](Self::run), which takes over the calling thread
/// until the convolver is dropped:
///
/// ```
/// use fft_convolver::ThreadedFFTConvolver;
///
/// let ir = vec![0.5_f32; 65_536];
/// let (mut convolver, mut worker) = ThreadedFFTConvolver::split(512, 8192, &ir).unwrap();
///
/// let handle = std::thread::spawn(move || {
///     // Set the thread up here: real-time priority, affinity, whatever your
///     // host needs. Then hand it to the worker.
///     worker.run();
/// });
///
/// let input = vec![1.0_f32; 512];
/// let mut output = vec![0.0_f32; 512];
/// convolver.process(&input, &mut output).unwrap();
///
/// drop(convolver); // ends run(), so the thread can be joined
/// handle.join().unwrap();
/// ```
///
/// To serve several convolvers from one thread, register that thread with each
/// worker and poll them with [`run_pending`](Self::run_pending) instead:
///
/// ```no_run
/// # use fft_convolver::TailWorker;
/// # fn example(mut workers: Vec<TailWorker<f32>>) {
/// for worker in &mut workers {
///     worker.set_waker(std::thread::current());
/// }
/// loop {
///     let mut worked = 0;
///     for worker in &mut workers {
///         worked += worker.run_pending();
///     }
///     if workers.iter().all(|worker| worker.is_disconnected()) {
///         break;
///     }
///     if worked == 0 {
///         std::thread::park();
///     }
/// }
/// # }
/// ```
///
/// The deadline is the same either way: a block handed over at the end of one
/// tail block period has to be convolved before the end of the next one. See
/// [`ThreadedFFTConvolver`] for what happens when it is not, and
/// [`missed_blocks`](ThreadedFFTConvolver::missed_blocks) for how to tell.
pub struct TailWorker<F: FftNum> {
    /// `None` when the impulse response is too short to have a tail stage, in
    /// which case every method is a no-op.
    inner: Option<WorkerInner<F>>,
}

struct WorkerInner<F: FftNum> {
    convolver: FFTConvolver<F>,
    input: Consumer<F>,
    output: Producer<F>,
    input_block: Vec<F>,
    output_block: Vec<F>,
    block_size: usize,
    shared: Arc<Shared>,
    /// Index of the block that will be read next.
    next_block: u64,
    /// Value of [`Shared::resync_at`] that has already been acted on.
    resync_applied: u64,
}

impl<F: FftNum> TailWorker<F> {
    /// A worker for a convolver that has no tail stage. Does nothing.
    fn idle() -> Self {
        Self { inner: None }
    }

    /// Convolves tail blocks until the convolver is dropped
    ///
    /// Registers the calling thread with [`set_waker`](Self::set_waker) and then
    /// alternates between doing the work that is ready and waiting for more.
    /// Returns once the convolver it belongs to has been dropped, or straight
    /// away if the impulse response has no tail stage.
    pub fn run(&mut self) {
        self.set_waker(thread::current());
        loop {
            self.run_pending();
            if self.is_disconnected() {
                return;
            }
            Semaphore::wait();
        }
    }

    /// Convolves every tail block that is ready and returns how many there were
    ///
    /// Never blocks. Use this to drive the worker from a thread that has other
    /// things to do, such as one serving several convolvers.
    pub fn run_pending(&mut self) -> usize {
        match self.inner.as_mut() {
            Some(inner) => inner.drain(),
            None => 0,
        }
    }

    /// Registers the thread to wake when a block becomes available
    ///
    /// Only needed when driving the worker with
    /// [`run_pending`](Self::run_pending) from a thread that parks itself;
    /// [`run`](Self::run) does it for you. The first registration wins, so a
    /// worker cannot be moved to a different thread and keep its wakeups.
    ///
    /// Without a registered thread nothing is lost, the worker simply is not
    /// woken and picks the work up the next time it is polled.
    pub fn set_waker(&self, thread: Thread) {
        if let Some(inner) = self.inner.as_ref() {
            inner.shared.semaphore.register(thread);
        }
    }

    /// Whether the convolver this worker belongs to has been dropped
    ///
    /// Also true when the impulse response has no tail stage, since there is
    /// nothing for the worker to do in that case either.
    pub fn is_disconnected(&self) -> bool {
        match self.inner.as_ref() {
            Some(inner) => inner.shared.shutdown.load(Ordering::Acquire),
            None => true,
        }
    }
}

impl<F: FftNum> std::fmt::Debug for TailWorker<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TailWorker")
            .field(
                "block_size",
                &self.inner.as_ref().map(|inner| inner.block_size),
            )
            .field("disconnected", &self.is_disconnected())
            .finish_non_exhaustive()
    }
}

impl<F: FftNum> WorkerInner<F> {
    /// Convolves every block that is ready and returns how many there were.
    ///
    /// Annotated as real-time even though this is not the audio thread: the
    /// work here has a deadline of one block period, so the same rules apply
    /// and RTSan should flag an allocation in the FFT path. The waiting happens
    /// in [`TailWorker::run`], outside of the checked region.
    #[nonblocking]
    fn drain(&mut self) -> usize {
        let mut processed = 0;
        loop {
            // Never drop a result: if there is no room, leave the input queued
            // and come back to it. Results have to stay one-to-one with the
            // blocks that were handed over, otherwise the audio thread cannot
            // tell which block a result belongs to.
            if self.output.slots() < self.block_size {
                break;
            }
            let Ok(chunk) = self.input.read_chunk(self.block_size) else {
                break;
            };
            let (first, second) = chunk.as_slices();
            self.input_block[..first.len()].copy_from_slice(first);
            self.input_block[first.len()..].copy_from_slice(second);
            chunk.commit_all();

            let resync_at = self.shared.resync_at.load(Ordering::Acquire);
            if resync_at > self.resync_applied && self.next_block >= resync_at {
                self.convolver.reset();
                self.resync_applied = resync_at;
            }
            self.next_block += 1;

            if self
                .convolver
                .process(&self.input_block, &mut self.output_block)
                .is_err()
            {
                self.output_block.fill(F::zero());
            }

            self.output
                .write_chunk_uninit(self.block_size)
                .expect("free slots were checked above")
                .fill_from_iter(self.output_block.iter().copied());
            self.shared
                .completed
                .store(self.next_block, Ordering::Release);

            processed += 1;
        }
        processed
    }
}

/// Everything that ties the audio thread to the worker.
///
/// Block accounting works with two monotonic counters. `pushed` is the number
/// of blocks handed to the worker, so the next one to hand over has index
/// `pushed`. `taken` is the number of results removed from the return ring, so
/// the oldest result still in it has index `taken`. The worker produces exactly
/// one result per block, in order, which makes the index of every result known
/// on both sides without tagging the samples.
struct Link<F: FftNum> {
    to_worker: Producer<F>,
    from_worker: Consumer<F>,
    shared: Arc<Shared>,
    /// Only set when the worker runs on a thread this crate spawned. With
    /// [`ThreadedFFTConvolver::split`] the caller owns the thread, and dropping
    /// the convolver just ends [`TailWorker::run`] instead of joining.
    worker: Option<JoinHandle<()>>,
    pushed: u64,
    taken: u64,
    /// Index of the first block whose result is still valid. Results from
    /// before a [`reset`](ThreadedFFTConvolver::reset) are stale and dropped.
    valid_from: u64,
    missed_blocks: u64,
}

impl<F: FftNum> Link<F> {
    #[blocking]
    fn shutdown(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.semaphore.post();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl<F: FftNum> Drop for Link<F> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl<F: FftNum> std::fmt::Debug for Link<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Link")
            .field("pushed", &self.pushed)
            .field("taken", &self.taken)
            .field("valid_from", &self.valid_from)
            .field("missed_blocks", &self.missed_blocks)
            .finish_non_exhaustive()
    }
}

/// ThreadedFFTConvolver
/// Two-stage FFT convolution with the tail stage on a worker thread.
///
/// [`TwoStageFFTConvolver`](crate::TwoStageFFTConvolver) computes the whole tail
/// on the call that completes a tail block, which makes that one call far more
/// expensive than its neighbours. This convolver keeps the head and the
/// transition stage on the audio thread, where they cost the same on every
/// call, and hands the tail block to a worker thread instead. The audio thread
/// only copies `tail_block_size` samples out and `tail_block_size` samples back
/// in once per tail block, so its cost per call becomes steady.
///
/// The worker has one full tail block period to do its work: the result of a
/// block handed over at the end of one period is not read until the end of the
/// next one. At 48 kHz with a tail block of 8192 samples that is 170 ms of wall
/// clock for a few hundred microseconds of work.
///
/// # Missed deadlines
///
/// The audio thread never waits for the worker. If a result has not arrived in
/// time, the tail contributes silence for that block period and
/// [`missed_blocks`](Self::missed_blocks) is incremented; the late result is
/// dropped when it turns up so the tail stays aligned with the input. Every
/// input block still reaches the worker in order, so the tail's history stays
/// intact and the output is exact again from the next block period onwards.
///
/// Only if the worker falls so far behind that the handoff backs up does the
/// tail get restarted, because blocks have to reach the worker in order and
/// skipping one would leave its history misaligned for good. The late part of
/// the response then fades back in over the length of the impulse response.
/// Reaching this point means the worker is several block periods behind, which
/// is a broken configuration rather than a hiccup.
///
/// Give the worker a real-time priority below the audio callback but above
/// normal threads; otherwise an unrelated busy thread can preempt it into a
/// missed deadline. The worker thread is named `fft-convolver-tail`.
///
/// # Real-time safety
///
/// [`process`](Self::process) and [`reset`](Self::reset) are real-time safe: no
/// allocations, no locks, and the only syscall is the wake that hands a block to
/// the worker, once per tail block. [`init`](Self::init),
/// [`init_default`](Self::init_default) and [`sync`](Self::sync) are not, and
/// neither is dropping the convolver, which joins the worker thread. Drop it
/// from the thread that created it, not from the audio callback.
///
/// Unlike the other two convolvers, this one has no `set_response`. Changing the
/// impulse response means re-running `init`, which spawns a new worker.
///
/// # Bring your own thread
///
/// [`init`](Self::init) spawns and owns the worker thread, which is the right
/// default for a single convolver. Two alternatives when it is not:
///
/// - [`init_with_setup`](Self::init_with_setup) still spawns the thread, but
///   runs a closure on it first, which is where a real-time priority has to be
///   requested from.
/// - [`split`](Self::split) spawns nothing and hands you a [`TailWorker`] to
///   place yourself, on a thread you have configured or on one that serves the
///   tails of several convolvers.
///
/// # Example
///
/// ```
/// use fft_convolver::ThreadedFFTConvolver;
///
/// let ir = vec![0.5_f32; 65_536];
/// let mut convolver = ThreadedFFTConvolver::default();
/// convolver.init_default(512, &ir).unwrap();
///
/// let input = vec![1.0_f32; 512];
/// let mut output = vec![0.0_f32; 512];
/// convolver.process(&input, &mut output).unwrap();
/// ```
#[derive(Debug)]
pub struct ThreadedFFTConvolver<F: FftNum> {
    ir_len: usize,
    head_block_size: usize,
    tail_block_size: usize,

    head_convolver: FFTConvolver<F>,
    tail_convolver0: FFTConvolver<F>,

    tail_input: Vec<F>,
    tail_input_fill: usize,

    tail_output0: Vec<F>,
    tail_precalculated0: Vec<F>,

    /// Tail result that is added to the output during the current block period.
    tail_precalculated: Vec<F>,

    /// Pre-allocated buffer to avoid allocations during process(), see
    /// [`TwoStageFFTConvolver`](crate::TwoStageFFTConvolver).
    processing_buffer: Vec<F>,

    precalculated_pos: usize,

    link: Option<Link<F>>,
}

impl<F: FftNum> Default for ThreadedFFTConvolver<F> {
    fn default() -> Self {
        Self {
            ir_len: Default::default(),
            head_block_size: Default::default(),
            tail_block_size: Default::default(),
            head_convolver: Default::default(),
            tail_convolver0: Default::default(),
            tail_input: Default::default(),
            tail_input_fill: Default::default(),
            tail_output0: Default::default(),
            tail_precalculated0: Default::default(),
            tail_precalculated: Default::default(),
            processing_buffer: Default::default(),
            precalculated_pos: Default::default(),
            link: Default::default(),
        }
    }
}

impl<F: FftNum> ThreadedFFTConvolver<F> {
    /// Initializes the convolver with an impulse response and starts the worker
    ///
    /// The impulse response is split the same way as in
    /// [`TwoStageFFTConvolver`](crate::TwoStageFFTConvolver): the head covers
    /// `ir[..tail_block_size]` and the transition stage
    /// `ir[tail_block_size..2 * tail_block_size]`, both at the head block size
    /// and both on the calling thread. The remainder is convolved at the tail
    /// block size on a worker thread, which is only spawned when the impulse
    /// response is longer than `2 * tail_block_size`.
    ///
    /// All memory allocation happens here, making subsequent processing
    /// real-time safe. Calling `init` again shuts the previous worker down.
    ///
    /// # Arguments
    ///
    /// * `head_block_size` - Block size for the head convolver (determines latency).
    ///   Will be rounded up to the next power of 2. Must be > 0.
    /// * `tail_block_size` - Block size for the tail convolver, and the deadline
    ///   the worker has to meet. Will be rounded up to the next power of 2.
    ///   Must be > 0. If smaller than `head_block_size`, the two values are
    ///   automatically swapped.
    /// * `impulse_response` - The impulse response to convolve with. Can be empty.
    ///
    /// # Returns
    ///
    /// Returns `BlockSizeZero` if either block size is 0, and `WorkerSpawn` if
    /// the worker thread could not be started.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::{ThreadedFFTConvolver, compute_tail_block_size};
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let head_block_size = 64;
    /// let tail_block_size = compute_tail_block_size(head_block_size, ir.len());
    ///
    /// let mut convolver = ThreadedFFTConvolver::default();
    /// convolver.init(head_block_size, tail_block_size, &ir).unwrap();
    /// ```
    pub fn init(
        &mut self,
        head_block_size: usize,
        tail_block_size: usize,
        impulse_response: &[F],
    ) -> Result<(), FFTConvolverError> {
        self.init_with_setup(head_block_size, tail_block_size, impulse_response, || {})
    }

    /// Initializes the convolver and runs `setup` on the worker thread
    ///
    /// Identical to [`init`](Self::init), except that `setup` is called on the
    /// worker thread before it starts convolving. This is where to put whatever
    /// the thread needs, most usefully a real-time priority: on Linux and macOS
    /// that has to be requested from the thread itself.
    ///
    /// Nothing is done to the thread's priority by default. The right value
    /// depends on the priority your host gave the audio callback, which this
    /// crate cannot know, and guessing too high would let the tail preempt the
    /// callback. The worker has a full tail block period to deliver, so an
    /// ordinary thread usually keeps up; use
    /// [`missed_blocks`](Self::missed_blocks) to find out whether yours does.
    ///
    /// `setup` is not run when the impulse response is short enough that no
    /// worker is needed.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::ThreadedFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let mut convolver = ThreadedFFTConvolver::default();
    /// convolver
    ///     .init_with_setup(64, 4096, &ir, || {
    ///         // e.g. audio_thread_priority::promote_current_thread_to_real_time(..)
    ///         // or a platform call of your own.
    ///     })
    ///     .unwrap();
    /// ```
    pub fn init_with_setup(
        &mut self,
        head_block_size: usize,
        tail_block_size: usize,
        impulse_response: &[F],
        setup: impl FnOnce() + Send + 'static,
    ) -> Result<(), FFTConvolverError> {
        let worker = self.build(head_block_size, tail_block_size, impulse_response)?;

        let Some(mut worker) = worker else {
            return Ok(());
        };

        let spawned = thread::Builder::new()
            .name("fft-convolver-tail".to_owned())
            .spawn(move || {
                setup();
                worker.run();
            });

        match spawned {
            Ok(handle) => {
                let link = self
                    .link
                    .as_mut()
                    .expect("a worker exists only together with its link");
                // Register before returning, so that no block can be handed
                // over before there is a thread to wake. Only the caller of
                // `split` can end up in that window, and only until it starts
                // its worker.
                link.shared.semaphore.register(handle.thread().clone());
                link.worker = Some(handle);
                Ok(())
            }
            Err(error) => {
                // Leaving the link in place would give a convolver whose tail
                // is never convolved, so drop everything instead.
                *self = Self::default();
                Err(error.into())
            }
        }
    }

    /// Initializes the convolver without starting a thread
    ///
    /// Returns the convolver together with its [`TailWorker`], which the caller
    /// has to drive. Use this to put the tail on a thread you set up yourself,
    /// or to run the tails of several convolvers on one thread. Everything else
    /// behaves exactly as after [`init`](Self::init).
    ///
    /// The worker returned for an impulse response with no tail stage does
    /// nothing, so there is no case to special-case.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::ThreadedFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let (mut convolver, mut worker) = ThreadedFFTConvolver::split(64, 4096, &ir).unwrap();
    ///
    /// let handle = std::thread::spawn(move || worker.run());
    ///
    /// let input = vec![1.0_f32; 256];
    /// let mut output = vec![0.0_f32; 256];
    /// convolver.process(&input, &mut output).unwrap();
    ///
    /// drop(convolver);
    /// handle.join().unwrap();
    /// ```
    pub fn split(
        head_block_size: usize,
        tail_block_size: usize,
        impulse_response: &[F],
    ) -> Result<(Self, TailWorker<F>), FFTConvolverError> {
        let mut convolver = Self::default();
        let worker = convolver.build(head_block_size, tail_block_size, impulse_response)?;
        Ok((convolver, worker.unwrap_or_else(TailWorker::idle)))
    }

    /// Sets everything up and returns the worker, if there is a tail stage.
    /// The caller decides whether to spawn a thread for it.
    fn build(
        &mut self,
        head_block_size: usize,
        tail_block_size: usize,
        impulse_response: &[F],
    ) -> Result<Option<TailWorker<F>>, FFTConvolverError> {
        if head_block_size == 0 || tail_block_size == 0 {
            return Err(FFTConvolverError::BlockSizeZero);
        }

        // Shuts the previous worker down, if there is one.
        *self = Self::default();

        self.head_block_size = next_power_of_2(head_block_size);
        self.tail_block_size = next_power_of_2(tail_block_size);

        if self.head_block_size > self.tail_block_size {
            std::mem::swap(&mut self.head_block_size, &mut self.tail_block_size);
        }

        self.ir_len = impulse_response.len();
        let ir_len = self.ir_len;

        if ir_len == 0 {
            return Ok(None);
        }

        let head_ir_len = ir_len.min(self.tail_block_size);
        self.head_convolver
            .init(self.head_block_size, &impulse_response[..head_ir_len])?;

        if ir_len > self.tail_block_size {
            let conv1_ir_len = (ir_len - self.tail_block_size).min(self.tail_block_size);
            self.tail_convolver0.init(
                self.head_block_size,
                &impulse_response[self.tail_block_size..self.tail_block_size + conv1_ir_len],
            )?;
            self.tail_output0 = vec![F::zero(); self.tail_block_size];
            self.tail_precalculated0 = vec![F::zero(); self.tail_block_size];
        }

        let mut worker = None;
        if ir_len > 2 * self.tail_block_size {
            self.tail_precalculated = vec![F::zero(); self.tail_block_size];
            let (link, tail_worker) =
                self.create_link(&impulse_response[2 * self.tail_block_size..])?;
            self.link = Some(link);
            worker = Some(tail_worker);
        }

        if !self.tail_precalculated0.is_empty() || !self.tail_precalculated.is_empty() {
            self.tail_input = vec![F::zero(); self.tail_block_size];
            self.processing_buffer = vec![F::zero(); self.tail_block_size];
        }

        self.tail_input_fill = 0;
        self.precalculated_pos = 0;

        Ok(worker)
    }

    /// Initializes the convolver with an automatically computed tail block size
    ///
    /// This is a convenience method that computes the tail block size using
    /// García's formula and then calls [`init`](Self::init). Note that the
    /// formula minimizes the total amount of work, not the deadline the worker
    /// has to meet; pass an explicit tail block size to `init` if you would
    /// rather give the worker a shorter or longer period.
    ///
    /// # Arguments
    ///
    /// * `head_block_size` - Block size for the head convolver (determines latency).
    ///   Will be rounded up to the next power of 2. Must be > 0.
    /// * `impulse_response` - The impulse response to convolve with. Can be empty.
    ///
    /// # Returns
    ///
    /// Returns `BlockSizeZero` if head_block_size is 0, and `WorkerSpawn` if
    /// the worker thread could not be started.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::ThreadedFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let mut convolver = ThreadedFFTConvolver::default();
    /// convolver.init_default(64, &ir).unwrap();
    /// ```
    pub fn init_default(
        &mut self,
        head_block_size: usize,
        impulse_response: &[F],
    ) -> Result<(), FFTConvolverError> {
        let tail_block_size = compute_tail_block_size(head_block_size, impulse_response.len());
        self.init(head_block_size, tail_block_size, impulse_response)
    }

    fn create_link(&self, tail_ir: &[F]) -> Result<(Link<F>, TailWorker<F>), FFTConvolverError> {
        let block_size = self.tail_block_size;

        let mut convolver = FFTConvolver::default();
        convolver.init(block_size, tail_ir)?;

        let (to_worker, input) = RingBuffer::new(RING_BLOCKS * block_size);
        let (output, from_worker) = RingBuffer::new(RING_BLOCKS * block_size);
        let shared = Arc::new(Shared::default());

        let worker = TailWorker {
            inner: Some(WorkerInner {
                convolver,
                input,
                output,
                input_block: vec![F::zero(); block_size],
                output_block: vec![F::zero(); block_size],
                block_size,
                shared: Arc::clone(&shared),
                next_block: 0,
                resync_applied: 0,
            }),
        };

        let link = Link {
            to_worker,
            from_worker,
            shared,
            worker: None,
            pushed: 0,
            taken: 0,
            valid_from: 0,
            missed_blocks: 0,
        };

        Ok((link, worker))
    }

    /// Convolves the input samples with the impulse response and outputs the result
    ///
    /// This is a real-time safe operation that performs no allocations. Internal
    /// buffering handles arbitrary sizes and ensures the output is always
    /// properly aligned with the input (zero latency except for processing
    /// time).
    ///
    /// Once per tail block this hands a block to the worker and picks up the
    /// previous result, which costs two copies of `tail_block_size` samples and
    /// one wake. If the worker missed its deadline, the tail contributes silence
    /// for that block period, see [`missed_blocks`](Self::missed_blocks).
    ///
    /// # Arguments
    ///
    /// * `input` - The input samples to convolve
    /// * `output` - Buffer to write the convolution result. Must have the same length as `input`.
    ///
    /// # Returns
    ///
    /// Returns `InputOutputLengthMismatch` if `input` and `output` have different lengths.
    /// Returns `Fft` error if an FFT operation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::ThreadedFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let mut convolver = ThreadedFFTConvolver::default();
    /// convolver.init_default(64, &ir).unwrap();
    ///
    /// let input = vec![1.0_f32; 256];
    /// let mut output = vec![0.0_f32; 256];
    /// convolver.process(&input, &mut output).unwrap();
    /// ```
    #[nonblocking]
    pub fn process(&mut self, input: &[F], output: &mut [F]) -> Result<(), FFTConvolverError> {
        if input.len() != output.len() {
            return Err(FFTConvolverError::InputOutputLengthMismatch);
        }
        self.head_convolver.process(input, output)?;

        if self.tail_input.is_empty() {
            return Ok(());
        }

        let len = input.len();
        let mut processed = 0;

        while processed < len {
            let remaining = len - processed;
            let processing =
                remaining.min(self.head_block_size - (self.tail_input_fill % self.head_block_size));

            let sum_begin = processed;
            let sum_end = processed + processing;

            // Add precalculated tail0 output
            if !self.tail_precalculated0.is_empty() {
                let precalc = &self.tail_precalculated0;
                let mut pos = self.precalculated_pos;
                #[allow(clippy::explicit_counter_loop)]
                for sample in &mut output[sum_begin..sum_end] {
                    *sample = *sample + precalc[pos];
                    pos += 1;
                }
            }

            // Add the tail result the worker delivered
            if !self.tail_precalculated.is_empty() {
                let precalc = &self.tail_precalculated;
                let mut pos = self.precalculated_pos;
                #[allow(clippy::explicit_counter_loop)]
                for sample in &mut output[sum_begin..sum_end] {
                    *sample = *sample + precalc[pos];
                    pos += 1;
                }
            }

            self.precalculated_pos += processing;

            // Buffer input for tail processing
            self.tail_input[self.tail_input_fill..self.tail_input_fill + processing]
                .copy_from_slice(&input[processed..processed + processing]);
            self.tail_input_fill += processing;

            // Process tail0 incrementally (every head_block_size samples)
            if !self.tail_precalculated0.is_empty()
                && self.tail_input_fill.is_multiple_of(self.head_block_size)
            {
                let block_offset = self.tail_input_fill - self.head_block_size;
                self.processing_buffer[..self.head_block_size].copy_from_slice(
                    &self.tail_input[block_offset..block_offset + self.head_block_size],
                );
                self.tail_convolver0.process(
                    &self.processing_buffer[..self.head_block_size],
                    &mut self.tail_output0[block_offset..block_offset + self.head_block_size],
                )?;
                if self.tail_input_fill == self.tail_block_size {
                    std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                }
            }

            if self.tail_input_fill == self.tail_block_size {
                self.exchange_tail_block();
                self.tail_input_fill = 0;
                self.precalculated_pos = 0;
            }

            processed += processing;
        }

        Ok(())
    }

    /// Picks up the result of the previous block and hands over the next one.
    ///
    /// Called at a tail block boundary, in the same place where
    /// [`TwoStageFFTConvolver`](crate::TwoStageFFTConvolver) swaps its tail
    /// buffers and runs the tail convolution inline.
    fn exchange_tail_block(&mut self) {
        let Some(link) = self.link.as_mut() else {
            return;
        };
        let block_size = self.tail_block_size;

        // The block handed over one period ago is the one due now.
        if link.pushed > 0 {
            let due = link.pushed - 1;

            // Anything older than that is a late result for a period the tail
            // has already contributed silence to. Drop it, so the results line
            // up with the input again.
            while link.taken < due && link.from_worker.slots() >= block_size {
                discard_block(&mut link.from_worker, block_size);
                link.taken += 1;
            }

            let arrived = link.taken == due && link.from_worker.slots() >= block_size;
            if arrived && due >= link.valid_from {
                let chunk = link
                    .from_worker
                    .read_chunk(block_size)
                    .expect("free slots were checked above");
                let (first, second) = chunk.as_slices();
                self.tail_precalculated[..first.len()].copy_from_slice(first);
                self.tail_precalculated[first.len()..].copy_from_slice(second);
                chunk.commit_all();
                link.taken += 1;
            } else {
                if arrived {
                    // Computed before the last reset, so no longer valid.
                    discard_block(&mut link.from_worker, block_size);
                    link.taken += 1;
                } else if due >= link.valid_from {
                    link.missed_blocks += 1;
                }
                self.tail_precalculated.fill(F::zero());
            }
        }

        // Hand the block that just filled up over to the worker.
        match link.to_worker.write_chunk_uninit(block_size) {
            Ok(chunk) => {
                chunk.fill_from_iter(self.tail_input.iter().copied());
                link.pushed += 1;
            }
            Err(_) => {
                // The worker is several blocks behind. Blocks have to reach it
                // in order, so instead of skipping one and leaving its history
                // misaligned for good, restart the tail from the next block.
                link.valid_from = link.pushed;
                link.shared.resync_at.store(link.pushed, Ordering::Release);
                link.missed_blocks += 1;
            }
        }

        link.shared.semaphore.post();
    }

    /// Clears the internal processing state while preserving the impulse response
    ///
    /// This real-time safe operation resets all internal buffers that store the
    /// convolution state, effectively removing any "history" or "tail" from
    /// previous processing. The impulse response configuration remains intact,
    /// so processing can continue immediately.
    ///
    /// This is useful when handling stream discontinuities such as:
    /// - Seeking in audio playback
    /// - Pause/resume operations with large time gaps
    /// - Switching between different audio sources
    ///
    /// The worker is not waited for. It is told to clear the tail state before
    /// the next block it receives, and results computed before that point are
    /// dropped when they arrive, so the audio thread does not stall.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::ThreadedFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let mut convolver = ThreadedFFTConvolver::default();
    /// convolver.init_default(64, &ir).unwrap();
    ///
    /// let input = vec![1.0_f32; 256];
    /// let mut output = vec![0.0_f32; 256];
    /// convolver.process(&input, &mut output).unwrap();
    ///
    /// convolver.reset();
    /// convolver.process(&input, &mut output).unwrap();
    /// ```
    #[nonblocking]
    pub fn reset(&mut self) {
        self.head_convolver.reset();
        self.tail_convolver0.reset();

        self.tail_input.fill(F::zero());
        self.tail_input_fill = 0;

        self.tail_output0.fill(F::zero());
        self.tail_precalculated0.fill(F::zero());
        self.tail_precalculated.fill(F::zero());

        self.processing_buffer.fill(F::zero());
        self.precalculated_pos = 0;

        if let Some(link) = self.link.as_mut() {
            // Everything the worker has produced so far belongs to the stream
            // before the reset.
            link.valid_from = link.pushed;
            link.shared.resync_at.store(link.pushed, Ordering::Release);
            link.shared.semaphore.post();
        }
    }

    /// Blocks until the worker has finished every block handed to it
    ///
    /// This is **not** real-time safe and must not be called from an audio
    /// callback. It makes this convolver deterministic, which is what tests
    /// want: call it after each `process` call of at most `tail_block_size`
    /// samples and no deadline can be missed.
    ///
    /// It is not the way to render offline. Waiting for the worker serialises
    /// the handoff, so it costs more than it saves when there is no real-time
    /// stream to keep up with;
    /// [`TwoStageFFTConvolver`](crate::TwoStageFFTConvolver) is faster there.
    ///
    /// Something has to be driving the worker for this to return. That is
    /// always the case after [`init`](Self::init), but after
    /// [`split`](Self::split) it means the caller has to be running
    /// [`TailWorker::run`] on another thread. If you drive the worker yourself
    /// with [`TailWorker::run_pending`], call that instead of this; it does the
    /// same work without blocking. Waiting for a worker that nobody runs would
    /// hang, so debug builds panic after a generous timeout rather than
    /// stopping silently.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::ThreadedFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 100_000];
    /// let mut convolver = ThreadedFFTConvolver::default();
    /// convolver.init_default(64, &ir).unwrap();
    ///
    /// let input = vec![1.0_f32; 256];
    /// let mut output = vec![0.0_f32; 256];
    /// for _ in 0..8 {
    ///     convolver.process(&input, &mut output).unwrap();
    ///     convolver.sync();
    /// }
    /// assert_eq!(convolver.missed_blocks(), 0);
    /// ```
    #[blocking]
    pub fn sync(&self) {
        let Some(link) = self.link.as_ref() else {
            return;
        };
        // At most two blocks are ever in flight and the rings hold four, so a
        // worker that is being driven can always make progress.
        #[cfg(debug_assertions)]
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

        while link.shared.completed.load(Ordering::Acquire) < link.pushed {
            #[cfg(debug_assertions)]
            assert!(
                std::time::Instant::now() < deadline,
                "sync() waited 5s for the tail worker; is anything running it? \
                 after split() the caller owns the thread, see TailWorker::run"
            );
            thread::yield_now();
        }
    }

    /// Number of block periods in which the tail contributed silence because the
    /// worker did not deliver in time
    ///
    /// Zero on a healthy system. A non-zero value means the worker did not get
    /// its work done within one tail block period, usually because it is not
    /// running at a high enough thread priority. The count is cumulative and
    /// survives [`reset`](Self::reset).
    pub fn missed_blocks(&self) -> u64 {
        self.link.as_ref().map_or(0, |link| link.missed_blocks)
    }

    /// The tail block size in use, after rounding up to the next power of 2
    ///
    /// This is the period the worker has to deliver a result in.
    pub fn tail_block_size(&self) -> usize {
        self.tail_block_size
    }
}

fn discard_block<F>(consumer: &mut Consumer<F>, block_size: usize) {
    if let Ok(chunk) = consumer.read_chunk(block_size) {
        chunk.commit_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TwoStageFFTConvolver;

    const HEAD: usize = 64;
    const TAIL: usize = 256;

    fn impulse_response(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.013).sin() * 0.5) / (1.0 + i as f32 * 0.001))
            .collect()
    }

    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * 0.017).sin() * 0.7 + (i as f32 * 0.031).cos() * 0.2)
            .collect()
    }

    /// Reference output of the single-threaded two-stage convolver.
    fn reference(head: usize, tail: usize, ir: &[f32], input: &[f32]) -> Vec<f32> {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        convolver.init(head, tail, ir).unwrap();
        let mut output = vec![0.0; input.len()];
        convolver.process(input, &mut output).unwrap();
        output
    }

    /// Drives the convolver in chunks of at most `tail_block_size`, waiting for
    /// the worker after each one. No deadline can be missed this way, so the
    /// output is deterministic and must match the reference exactly.
    fn process_synced(
        convolver: &mut ThreadedFFTConvolver<f32>,
        input: &[f32],
        chunk: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        let mut pos = 0;
        while pos < input.len() {
            let end = (pos + chunk).min(input.len());
            convolver
                .process(&input[pos..end], &mut output[pos..end])
                .unwrap();
            convolver.sync();
            pos = end;
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tolerance,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    /// The smallest end-to-end run that still spawns a worker, hands blocks
    /// over, picks results up and shuts down again. Kept small enough to stay
    /// in the Miri run, where the rest of the threaded tests are too slow: this
    /// is the one that has to catch undefined behaviour in the handoff.
    #[test]
    fn miri_smoke() {
        let head = 4;
        let tail = 16;
        let mut ir = vec![0.0_f32; 100];
        ir[0] = 0.5;
        ir[40] = 0.25;

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(head, tail, &ir).unwrap();
        assert!(convolver.link.is_some());

        let mut input = vec![0.0_f32; 6 * tail];
        input[1] = 1.0;
        let output = process_synced(&mut convolver, &input, tail);

        assert!((output[1] - 0.5).abs() < 1e-5, "head: {}", output[1]);
        assert!((output[41] - 0.25).abs() < 1e-5, "worker: {}", output[41]);
        assert_eq!(convolver.missed_blocks(), 0);

        convolver.reset();
        let output = process_synced(&mut convolver, &input, tail);
        assert!(
            (output[41] - 0.25).abs() < 1e-5,
            "after reset: {}",
            output[41]
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn init_splits_the_response_and_starts_a_worker() {
        let ir = impulse_response(4000);
        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        assert_eq!(convolver.head_block_size, HEAD);
        assert_eq!(convolver.tail_block_size, TAIL);
        assert_eq!(convolver.tail_input.len(), TAIL);
        assert_eq!(convolver.tail_precalculated.len(), TAIL);
        assert!(convolver.link.is_some());
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn no_worker_without_a_tail() {
        // Shorter than 2 * tail_block_size, so there is nothing to offload.
        let ir = impulse_response(300);
        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        assert!(convolver.link.is_none());
        assert!(convolver.tail_precalculated.is_empty());
        assert_eq!(convolver.missed_blocks(), 0);

        let input = signal(1024);
        let output = process_synced(&mut convolver, &input, TAIL);
        assert_close(&output, &reference(HEAD, TAIL, &ir, &input), 1e-5);
    }

    #[test]
    fn init_block_size_zero_returns_error() {
        let ir = impulse_response(1000);
        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        assert!(matches!(
            convolver.init(0, TAIL, &ir),
            Err(FFTConvolverError::BlockSizeZero)
        ));
        assert!(matches!(
            convolver.init(HEAD, 0, &ir),
            Err(FFTConvolverError::BlockSizeZero)
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn process_mismatched_lengths_returns_error() {
        let ir = impulse_response(4000);
        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        let input = vec![1.0_f32; 64];
        let mut output = vec![0.0_f32; 128];
        assert!(matches!(
            convolver.process(&input, &mut output),
            Err(FFTConvolverError::InputOutputLengthMismatch)
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn matches_the_single_threaded_convolver() {
        let ir = impulse_response(4000);
        let input = signal(4096);

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();
        let output = process_synced(&mut convolver, &input, TAIL);

        assert_close(&output, &reference(HEAD, TAIL, &ir, &input), 1e-5);
        assert_eq!(
            convolver.missed_blocks(),
            0,
            "synced processing must not miss a deadline"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn matches_with_varying_chunk_sizes() {
        let ir = impulse_response(4000);
        let input = signal(2048);
        let expected = reference(HEAD, TAIL, &ir, &input);

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        let mut output = vec![0.0; input.len()];
        let mut pos = 0;
        // Chunk sizes stay at or below the tail block size so that a single
        // process() call never crosses two block boundaries.
        for &chunk in [1_usize, 7, 64, 100, 3, 256, 13, 200, 255, 128]
            .iter()
            .cycle()
        {
            let end = (pos + chunk).min(input.len());
            convolver
                .process(&input[pos..end], &mut output[pos..end])
                .unwrap();
            convolver.sync();
            pos = end;
            if pos == input.len() {
                break;
            }
        }

        assert_close(&output, &expected, 1e-5);
        assert_eq!(convolver.missed_blocks(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn zero_latency() {
        let mut ir = vec![0.0_f32; 4000];
        ir[0] = 0.5;
        ir[1] = 0.3;

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        let mut input = vec![0.0_f32; TAIL];
        input[0] = 1.0;
        let output = process_synced(&mut convolver, &input, TAIL);

        assert!((output[0] - 0.5).abs() < 1e-5, "got {}", output[0]);
        assert!((output[1] - 0.3).abs() < 1e-5, "got {}", output[1]);
    }

    /// The worker only contributes from `2 * tail_block_size` onwards, so an
    /// impulse response that is an impulse at that offset can only come out
    /// right if the handoff is aligned correctly.
    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn worker_contribution_is_aligned() {
        let offset = 2 * TAIL + 5;
        let mut ir = vec![0.0_f32; 4 * TAIL];
        ir[offset] = 1.0;

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        let len = 8 * TAIL;
        let mut input = vec![0.0_f32; len];
        input[3] = 1.0;
        let output = process_synced(&mut convolver, &input, TAIL);

        for (i, &sample) in output.iter().enumerate() {
            let expected = if i == offset + 3 { 1.0 } else { 0.0 };
            assert!(
                (sample - expected).abs() < 1e-5,
                "at {i}: got {sample}, expected {expected}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn reset_matches_a_fresh_convolver() {
        let ir = impulse_response(4000);
        let history = signal(2048);
        let input = signal(2048);

        let mut used = ThreadedFFTConvolver::<f32>::default();
        used.init(HEAD, TAIL, &ir).unwrap();
        process_synced(&mut used, &history, TAIL);
        // Synced, so a finished result for the block before the reset is
        // waiting in the return ring and has to be dropped as stale.
        used.sync();
        used.reset();
        let after_reset = process_synced(&mut used, &input, TAIL);

        let mut fresh = ThreadedFFTConvolver::<f32>::default();
        fresh.init(HEAD, TAIL, &ir).unwrap();
        let from_fresh = process_synced(&mut fresh, &input, TAIL);

        assert_close(&after_reset, &from_fresh, 1e-5);
        assert_eq!(used.missed_blocks(), 0, "a reset is not a missed deadline");
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn reset_preserves_the_configuration() {
        let ir = impulse_response(4000);
        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        let input = signal(1024);
        process_synced(&mut convolver, &input, TAIL);
        convolver.reset();

        assert_eq!(convolver.ir_len, 4000);
        assert_eq!(convolver.head_block_size, HEAD);
        assert_eq!(convolver.tail_block_size(), TAIL);
        assert!(convolver.link.is_some());
    }

    /// Pushes far more work at the convolver than a callback ever would: single
    /// calls crossing forty block boundaries back to back, with the tail stage
    /// carrying two orders of magnitude more work per block period than the head
    /// stages. The worker cannot keep up, so deadlines are missed and it ends up
    /// far enough behind for the tail to be restarted.
    ///
    /// The output must stay finite throughout, the misses must stop as soon as
    /// the pace is sane again, and the output must become exact once the tail
    /// has had a full impulse response worth of input to rebuild itself from.
    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri")]
    fn recovers_from_overload() {
        let head = 64;
        let tail = 256;
        let ir = impulse_response(20_000);

        let burst = signal(40 * tail);
        let recovery = signal(140 * tail);

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(head, tail, &ir).unwrap();

        // The reference sees the same input, so its history stays comparable.
        let mut expected_convolver = TwoStageFFTConvolver::<f32>::default();
        expected_convolver.init(head, tail, &ir).unwrap();

        let mut overloaded = vec![0.0; burst.len()];
        convolver.process(&burst, &mut overloaded).unwrap();
        let mut expected = vec![0.0; burst.len()];
        expected_convolver.process(&burst, &mut expected).unwrap();

        assert!(
            overloaded.iter().all(|sample| sample.is_finite()),
            "overload must not produce garbage"
        );
        assert!(
            convolver.missed_blocks() > 0,
            "the burst is meant to overload the worker"
        );

        // Back to the pace a callback would use.
        let mut output = process_synced(&mut convolver, &recovery[..4 * tail], tail);
        let settled = convolver.missed_blocks();
        output.extend(process_synced(&mut convolver, &recovery[4 * tail..], tail));

        let mut expected = vec![0.0; recovery.len()];
        expected_convolver
            .process(&recovery, &mut expected)
            .unwrap();

        assert_eq!(
            convolver.missed_blocks(),
            settled,
            "no deadline may be missed once the worker is given its block period"
        );

        // A restarted tail has to be fed a full impulse response before it can
        // match again; from there on the output must be exact.
        let rebuilt = ir.len().next_multiple_of(tail) + tail;
        assert_close(&output[rebuilt..], &expected[rebuilt..], 1e-4);
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn split_matches_the_spawning_constructor() {
        let ir = impulse_response(4000);
        let input = signal(2048);

        let (mut convolver, mut worker) = ThreadedFFTConvolver::<f32>::split(HEAD, TAIL, &ir)
            .expect("split must not fail for a valid configuration");
        let handle = std::thread::spawn(move || worker.run());

        let output = process_synced(&mut convolver, &input, TAIL);

        assert_close(&output, &reference(HEAD, TAIL, &ir, &input), 1e-5);
        assert_eq!(convolver.missed_blocks(), 0);

        // Dropping the convolver has to end run(), or this join hangs.
        drop(convolver);
        handle.join().unwrap();
    }

    /// The worker does not need a thread of its own at all: driving it by hand
    /// between process() calls gives the same output and never blocks.
    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn split_can_be_driven_without_a_thread() {
        let ir = impulse_response(4000);
        let input = signal(4 * TAIL);

        let (mut convolver, mut worker) =
            ThreadedFFTConvolver::<f32>::split(HEAD, TAIL, &ir).unwrap();

        let mut output = vec![0.0; input.len()];
        let mut pos = 0;
        while pos < input.len() {
            let end = pos + TAIL;
            convolver
                .process(&input[pos..end], &mut output[pos..end])
                .unwrap();
            // Stands in for the callback period: the worker gets its turn
            // between one block boundary and the next.
            worker.run_pending();
            pos = end;
        }

        assert_close(&output, &reference(HEAD, TAIL, &ir, &input), 1e-5);
        assert_eq!(convolver.missed_blocks(), 0);
        assert!(!worker.is_disconnected());

        drop(convolver);
        assert!(worker.is_disconnected());
    }

    /// One thread serving several convolvers, which is the reason the split
    /// exists: eight instances should not mean eight threads.
    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn one_thread_can_serve_several_convolvers() {
        let ir = impulse_response(4000);
        let input = signal(2048);
        let expected = reference(HEAD, TAIL, &ir, &input);

        let mut convolvers = Vec::new();
        let mut workers = Vec::new();
        for _ in 0..8 {
            let (convolver, worker) = ThreadedFFTConvolver::<f32>::split(HEAD, TAIL, &ir).unwrap();
            convolvers.push(convolver);
            workers.push(worker);
        }

        let handle = std::thread::spawn(move || {
            for worker in &mut workers {
                worker.set_waker(std::thread::current());
            }
            loop {
                let mut worked = 0;
                for worker in &mut workers {
                    worked += worker.run_pending();
                }
                if workers.iter().all(|worker| worker.is_disconnected()) {
                    return;
                }
                if worked == 0 {
                    std::thread::park_timeout(std::time::Duration::from_millis(1));
                }
            }
        });

        for convolver in &mut convolvers {
            let output = process_synced(convolver, &input, TAIL);
            assert_close(&output, &expected, 1e-5);
            assert_eq!(convolver.missed_blocks(), 0);
        }

        drop(convolvers);
        handle.join().unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn split_without_a_tail_yields_an_idle_worker() {
        // Too short for a tail stage, so there is nothing for a worker to do.
        let ir = impulse_response(300);
        let input = signal(1024);

        let (mut convolver, mut worker) =
            ThreadedFFTConvolver::<f32>::split(HEAD, TAIL, &ir).unwrap();

        assert!(worker.is_disconnected());
        assert_eq!(worker.run_pending(), 0);
        // Must return immediately rather than park forever.
        worker.run();

        let output = process_synced(&mut convolver, &input, TAIL);
        assert_close(&output, &reference(HEAD, TAIL, &ir, &input), 1e-5);
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn setup_runs_on_the_worker_thread() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ir = impulse_response(4000);
        let ran = Arc::new(AtomicBool::new(false));
        let elsewhere = Arc::new(AtomicBool::new(false));

        let flag = Arc::clone(&ran);
        let on_worker = Arc::clone(&elsewhere);
        let caller = std::thread::current().id();

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver
            .init_with_setup(HEAD, TAIL, &ir, move || {
                flag.store(true, Ordering::Release);
                on_worker.store(std::thread::current().id() != caller, Ordering::Release);
            })
            .unwrap();

        // The setup runs before any convolution, so it has happened by the time
        // the first block comes back.
        let input = signal(2 * TAIL);
        process_synced(&mut convolver, &input, TAIL);

        assert!(ran.load(Ordering::Acquire), "setup must run");
        assert!(
            elsewhere.load(Ordering::Acquire),
            "setup must run on the worker, not on the caller"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn setup_is_skipped_when_there_is_no_worker() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ir = impulse_response(300);
        let ran = Arc::new(AtomicBool::new(false));

        let flag = Arc::clone(&ran);
        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver
            .init_with_setup(HEAD, TAIL, &ir, move || {
                flag.store(true, Ordering::Release);
            })
            .unwrap();

        let input = signal(1024);
        process_synced(&mut convolver, &input, TAIL);

        assert!(!ran.load(Ordering::Acquire), "no worker, no setup");
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn re_init_replaces_the_worker() {
        let short = impulse_response(4000);
        let long = impulse_response(9000);
        let input = signal(2048);

        let mut convolver = ThreadedFFTConvolver::<f32>::default();
        convolver.init(HEAD, TAIL, &long).unwrap();
        process_synced(&mut convolver, &input, TAIL);

        convolver.init(HEAD, TAIL, &short).unwrap();
        let output = process_synced(&mut convolver, &input, TAIL);

        assert_close(&output, &reference(HEAD, TAIL, &short, &input), 1e-5);
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn dropping_shuts_the_worker_down() {
        let ir = impulse_response(4000);
        for _ in 0..8 {
            let mut convolver = ThreadedFFTConvolver::<f32>::default();
            convolver.init(HEAD, TAIL, &ir).unwrap();
            // Dropped mid-flight, with a block still queued for the worker.
            let input = signal(TAIL);
            let mut output = vec![0.0; TAIL];
            convolver.process(&input, &mut output).unwrap();
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "too slow under miri, see miri_smoke")]
    fn works_with_f64() {
        let ir: Vec<f64> = (0..4000).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let input: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();

        let mut convolver = ThreadedFFTConvolver::<f64>::default();
        convolver.init(HEAD, TAIL, &ir).unwrap();

        let mut expected_convolver = TwoStageFFTConvolver::<f64>::default();
        expected_convolver.init(HEAD, TAIL, &ir).unwrap();
        let mut expected = vec![0.0; input.len()];
        expected_convolver.process(&input, &mut expected).unwrap();

        let mut output = vec![0.0; input.len()];
        let mut pos = 0;
        while pos < input.len() {
            let end = (pos + TAIL).min(input.len());
            convolver
                .process(&input[pos..end], &mut output[pos..end])
                .unwrap();
            convolver.sync();
            pos = end;
        }

        assert_eq!(output.len(), expected.len());
        for (i, (&a, &e)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (a - e).abs() < 1e-9,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }
}
