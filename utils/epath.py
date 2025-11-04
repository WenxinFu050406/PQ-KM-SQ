"""
enhanced Path class

"""
import os
import sys
import glob
from pathlib import Path, PosixPath




def get_parent_stem_suffix(path):
    path = Path(path)
    # extract the rightmost suffix
    parent, stem, suffix = path.parent, path.stem, path.suffix
    return parent, stem, suffix


def replace_suffix(path, new_suffix):
    new_suffix = str(new_suffix)
    parent,stem, suffix = get_parent_stem_suffix(path)
    if new_suffix.startswith("."):
        new_suffix = new_suffix[1:]

    bsn = '.'.join([stem, new_suffix])
    new_path = Path(parent).joinpath(bsn)
    return str(new_path)


class EPath(PosixPath):
    """
    q: quick access method 
    shortcut method name: name through _ + initials
        - eg: EPath._rp(parent)
    c: convenient method usage
    """
    def assert_exists(self):
        assert self.exists(), f'{self} doesn\'t exist'
        return self
    _ae = assert_exists

    def replace_parent(self, new_parents):
        """
        replaces parent by new_parents

        """
        path = os.path.join(str(new_parents), self.name)
        return EPath(path)
    _rp = replace_parent

    def replace_suffix(self, new_suffix):
        """
        replaces last suffix of path, if no suffix is found, it will add one

        """
        return EPath(replace_suffix(self, new_suffix))
    _rs = replace_suffix

    def preplace(self, parent=None, suffix=None):
        """
        replace parent and / or suffix
        """
        path = EPath(self)
        if parent is not None:
            path = path.replace_parent(parent)
        if suffix is not None:
            path = path.replace_suffix(suffix)
        return path

    def add2stem(self, ssuffix, after=True, sep="_"):
        """
        add a stem suffix, i.e. some extra stem information before the extension

        :attribute: path_str
        :rtype: EPath
        :returns: EPath object with an extra text before or after stem
        # add an extra suffix after the current stem
        :Example:
        >>> path = EPath("/dirA/dirB/myfile.ext1")
        >>> path.add_stem("p1_p2_param3", after=True)
        /dirA/dirB/p1_p2_param3_myfile.ext1
        """
        if after:
            basename = "".join([self.stem, sep, ssuffix, self.suffix])
        else:
            basename = "".join([ssuffix, sep, self.stem, self.suffix])
        path = os.path.join(self.parent, basename)
        return EPath(path)
    _as = add2stem

    def fsize(self):
        return self.stat().st_size

    def pglob(self, pattern, sort=True):
        """
        returns a list of all files matching the pattern as a list of EPath objects

        :param pattern: pattern to match
        :type pattern: str
        :param sort: sort the list
        :type sort: bool
        :param epath: return the EPath object
        :type epath: bool
        """
        files = glob.glob(str(self / pattern))
        if sort is True:
            files.sort()
        files = [EPath(f) for f in files]
        return files
    
    def pglob_jpg(self):
        return self.pglob('*.jpg')
    
    def pglob_json(self):
        return self.pglob('*.json')

    def qmkdir(self):
        """
        quick mkdir shortcut to create at path definition
        usage: path = path.qmkdir()
        """
        self.mkdir(exist_ok=True)
        return self

    @property
    def s(self):
        return str(self)

    # __str__ = s
    # __repr__ = __str__