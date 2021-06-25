""" This extends the python multiprocessing framework to add some additional functionality. This is required for
passing seeded random generators and to increase the chunk size that is passed between processes.

"""

import multiprocessing as mp
from multiprocessing import util  # noqa
from multiprocessing.pool import (  # noqa
    RUN,
    ExceptionWithTraceback,
    MapResult,
    Pool,
    _helper_reraises_exception,
)
from operator import methodcaller
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from eta_utility import get_logger

log = get_logger("eta_x.agents")
cpu_count = mp.cpu_count


def pool_worker(
    inqueue: mp.SimpleQueue,
    outqueue: mp.SimpleQueue,
    initializer: Optional[Callable] = None,
    initargs: Tuple[Any] = (),
    maxtasks: Optional[int] = None,
    wrap_exception: bool = False,
) -> None:
    """Worker function for process pool members. Runs in a loop and waits until new tasks arrive.
    Pass None through the inqueue to stop interrupt the loop.

    :param inqueue: Queue for incoming tasks
    :param outqueue: Queue for outgoing results
    :param initializer: Function to be called upon process startup
    :param initargs: Arguments for initilizer function
    :param maxtasks: Maximum number of tasks to perform before terminating process
    :param wrap_exception: Wrap exception tracebacks
    :return:
    """
    if (maxtasks is not None) and not (isinstance(maxtasks, int) and maxtasks >= 1):
        raise AssertionError(f"Maxtasks {maxtasks!r} is not valid")

    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, "_writer"):
        inqueue._writer.close()  # noqa
        outqueue._reader.close()  # noqa

    if initializer is not None:
        initializer(*initargs)

    completed = 0
    while maxtasks is None or (maxtasks and completed < maxtasks):
        try:
            task = get()
        except (EOFError, OSError):
            util.debug("worker got EOFError or OSError -- exiting")
            break

        if task is None:
            util.debug("worker got sentinel -- exiting")
            break

        job, num, func, iterable, method, args, kwargs = task
        result = (False, ())
        try:

            if method in {"return", "return_rng"}:
                if not callable(func):
                    caller = methodcaller(func, *args, **kwargs)
                    result = (True, list(caller(i) for i in iterable))
                else:
                    result = (True, list(func(i, *args, **kwargs) for i in iterable))

            elif method in {"modify", "modify_rng"}:
                if not callable(func):
                    caller = methodcaller(func, *args, **kwargs)
                    for i in iterable:
                        caller(i)
                    result = (True, iterable)
            else:
                if func is _helper_reraises_exception:
                    func(num, *args, **kwargs)
                else:
                    raise ValueError(f"Invalid mapping method specified: {method}")

        except Exception as e:
            if wrap_exception and func is not _helper_reraises_exception:
                e = ExceptionWithTraceback(e, e.__traceback__)
            result = (False, e)

        try:
            put((job, num, result))
        except Exception as e:
            put(
                (
                    False,
                    f"Worker failed. Possibly due to an error while encoding the result: {e}",
                )
            )

        # Reset all variables to ensure that every loop starts fresh
        job = num = func = iterable = method = args = kwargs = task = result = None  # noqa
        completed += 1


class ProcessPool(Pool):
    """Reimplementation of the multiprocessing.Pool helper. This offers some additional flexibility with regard
    to handling random number generators and differentiation between function and method calls.

    The implementation only has a working "map" function. Other functions of multiprocessing.Pool will raise
    NotImplementedError.

    :param seed_sequence: Numpy random number seed sequence to use within processes
    :param args: Arguments for the multiprocessing.Pool class
    :param kwargs: Keyword parameters for the multiprocessing.Pool class.
    """

    def __init__(self, *args: Any, seed_sequence: Optional[np.random.SeedSequence] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._rngs: List[np.random.BitGenerator] = self._setup_rngs(seed_sequence)

    def _repopulate_pool(self) -> None:
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(self._processes - len(self._pool)):  # noqa
            w = self.Process(
                target=pool_worker,  # noqa
                args=(
                    self._inqueue,
                    self._outqueue,  # noqa
                    self._initializer,  # noqa
                    self._initargs,
                    self._maxtasksperchild,  # noqa
                    self._wrap_exception,
                ),  # noqa
            )
            self._pool.append(w)  # noqa
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            util.debug("added worker")

    def _guarded_task_generation(self, result_job, func, iterable, method, args, kwargs) -> None:
        try:
            i = -1
            for i, iter_ in enumerate(iterable):
                if method in {"modify_rng", "return_rng"}:
                    kwargs["rng"] = self._rngs[i]
                yield result_job, i, func, iter_[-1], method, args, kwargs

        except Exception as e:
            yield result_job, i + 1, _helper_reraises_exception, (e,), "", (), {}  # noqa

    def map(
        self,
        func: Union[str, Callable],
        iterable: Iterable[Any],
        chunksize: Optional[int] = None,
        *,  # noqa
        args: Iterable = (),
        kwargs: Optional[Mapping] = None,
        method: str = "return",
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
    ) -> Sequence:
        """Perform parallel mapping operation with func on iterable. Uses arguments and kwarguments for the
        function call.

        :param func: A callable object to apply to all elements of iterable
        :param iterable: An iterable object
        :param chunksize: Chunksize for spreading workload between processes
        :param args: List or tuple of additional arguments for the function
        :param kwargs: Dictionary or namedtuple of additional keyword arguments for the function
        :param method: Either return of modify. Determines whether return values are used or the object is modified
                           in place. Append "_rng" to pass a pseudo random number generator to the process
        :param Callable callback: Called after processing
        :param Callable error_callback: Called after error
        :return: Modified Sequence
        """
        assert method in {"return", "modify", "return_rng", "modify_rng"}
        if self._state != RUN:  # noqa
            raise ValueError("Pool not running")
        if not hasattr(iterable, "__len__"):
            iterable = list(iterable)
        if len(iterable) <= 0:
            raise ValueError("Iterable has no elements.")
        if not hasattr(args, "__len__"):
            raise ValueError("Arguments must be given as an iterable.")
        if kwargs is None:
            kwargs = {}
        elif not hasattr(kwargs, "items"):
            raise ValueError("Keywords must be given as a dictionary.")

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool))  # noqa
            if extra:
                chunksize += 1
        if len(iterable) == 0:
            chunksize = 0

        task_batches = Pool._get_tasks(func, iterable, chunksize)  # noqa
        result = MapResult(
            self._cache,
            chunksize,
            len(iterable),
            callback,
            error_callback=error_callback,
        )  # noqa

        self._taskqueue.put(  # noqa
            (
                self._guarded_task_generation(result._job, func, task_batches, method, args, kwargs),  # noqa
                None,
            )
        )
        return_result = result.get()

        if method in {"return_rng", "modify_rng"}:
            for gen in self._rngs:
                gen.bit_generator.advance(chunksize * 10000)

        return return_result

    def _setup_rngs(self, seed_sequence: np.random.SeedSequence) -> List[np.random.BitGenerator]:
        """Take a seed sequence from numpy and set up corresponding, non overlapping random number generators
            for each processor

        :param np.random.SeedSequence seed_sequence: Original seed sequence
        :return: list of numpy random number generators
        """

        if seed_sequence is not None:
            seeds = seed_sequence.spawn(len(self._pool))  # noqa
            rngs = []

            for seed in seeds:
                rngs.append(np.random.default_rng(seed))

        else:
            rngs = None

        return rngs

    def imap(self, *args, **kwargs) -> None:
        """Imap is not implemented in this version of the process pool."""
        raise NotImplementedError("Imap is not implemented in this version of the process pool.")

    def imap_unordered(self, *args, **kwargs) -> None:
        """Unordered imap is not implemented in this version of the process pool."""

        raise NotImplementedError("Unordered imap is not implemented in this version of the process pool.")

    def apply_async(self, *args, **kwargs) -> None:
        """Asynchronous apply is not implemented in this version of the process pool."""
        raise NotImplementedError("Asynchronous apply is not implemented in this version of the process pool.")

    def _map_async(self, *args, **kwargs) -> None:
        """This is not implemented. Since it's the backend of all mapping functions, none of those will work."""
        raise NotImplementedError("Asynchronous map is not implemented in this version of the process pool.")
