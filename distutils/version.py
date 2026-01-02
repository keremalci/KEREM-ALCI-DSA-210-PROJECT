from packaging.version import Version

class LooseVersion:
    """Minimal LooseVersion shim using packaging.Version for simple comparisons.
    This is a small compatibility shim to satisfy imports that expect
    `distutils.version.LooseVersion` on Python versions where distutils
    is not available in the standard library (e.g. Python 3.14+).
    """
    def __init__(self, v):
        try:
            self.version = Version(str(v))
        except Exception:
            # fall back to using the raw string if packaging can't parse
            self.version = str(v)

    def _other_version(self, other):
        if isinstance(other, LooseVersion):
            return other.version
        try:
            return Version(str(other))
        except Exception:
            return str(other)

    def __lt__(self, other):
        o = self._other_version(other)
        return self.version < o

    def __le__(self, other):
        o = self._other_version(other)
        return self.version <= o

    def __gt__(self, other):
        o = self._other_version(other)
        return self.version > o

    def __ge__(self, other):
        o = self._other_version(other)
        return self.version >= o

    def __eq__(self, other):
        o = self._other_version(other)
        return self.version == o

    def __ne__(self, other):
        return not self.__eq__(other)
    