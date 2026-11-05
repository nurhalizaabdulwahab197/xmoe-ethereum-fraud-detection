"""Retry transient Windows file-sharing errors without touching ML RNG states."""
import errno
import json
from pathlib import Path
import time
import uuid
import warnings


def replace_retry(source, target, attempts=40):
    source, target = Path(source), Path(target)
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except OSError as error:
            transient = getattr(error, 'winerror', None) in (5, 32, 33) or error.errno in (errno.EACCES, errno.EPERM, errno.EBUSY)
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(min(0.01 * (1.6**attempt), 0.25))


def atomic_json(path, value, required=True):
    path = Path(path)
    # UUID uses OS entropy, not Python/NumPy/Torch PRNG state used in training.
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        replace_retry(temporary, path)
        return True
    except OSError as error:
        if required:
            raise
        warnings.warn('Noncritical progress update skipped after file-I/O retries: ' + str(error), RuntimeWarning)
        return False
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
