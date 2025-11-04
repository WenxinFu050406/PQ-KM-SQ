import os
import sys
import glob
import json
from typing import List
# from tqdm import tqdm



class JSONIO:
    def __init__(self, fname) -> None:
        self.fname = fname
    
    def read(self):
        with open(self.fname, 'r') as fd:
            data = json.load(fd)
        return data
    
    def write(self, data):
        with open(self.fname, 'w') as fd:
            json.dump(data, fd, indent=2, sort_keys=True)

try:
    import orjson
    class ORJSONIO:
        def __init__(self, fname) -> None:
            self.fname = fname
        
        def read(self):
            with open(self.fname, 'rb') as fd:
                data = orjson.loads(fd.read())
            return data
        
        def write(self, data):
            with open(self.fname, 'wb') as fd:
                fd.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
except:
    orjson = json
    ORJSONIO = JSONIO




class JSONAccumulator:
    def __init__(self, directory, pattern='*.json') -> None:
        """
        directory: contains the json files to accumulate
        pattern: a basename pattern to glob particular json
        """
        self.directory = directory
        self.pattern = pattern
    
    def glob(self):
        return list(glob.glob(os.path.join(self.directory, self.pattern)))

    def accumulate(self, output):
        """
        output: json output file name
        """
        json_files = self.glob()
        print(f'Accumulating {len(json_files)} json files in {self.directory}')
        accumulated = []
        for f in tqdm(json_files):

            content = ORJSONIO(f).read()
            if isinstance(content, list):
                accumulated.extend(content)
            else:
                accumulated.append(content)
        ORJSONIO(output).write(accumulated)