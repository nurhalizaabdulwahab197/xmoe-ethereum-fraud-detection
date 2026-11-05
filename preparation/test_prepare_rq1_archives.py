import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from prepare_rq1 import extract_graphs


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, nested=True, ambiguous=False, missing=False, wrong_size=False):
        source=root/'source.zip'
        payload=b'primitive graph fixture'
        with zipfile.ZipFile(source,'w') as outer:
            if nested:
                stream=io.BytesIO()
                with zipfile.ZipFile(stream,'w') as inner:
                    inner.writestr('xlabel_normal.pickle' if not missing else 'other.txt',payload)
                outer.writestr('folder/normal_pkl.zip',stream.getvalue())
            if not nested or ambiguous:
                outer.writestr('../../xlabel_normal.pickle',payload)
        raw=root/'raw';raw.mkdir()
        bounds={'normal':{'file':'normal/xlabel_normal.pickle',
                         'file_bytes':len(payload)+(1 if wrong_size else 0)}}
        return source,raw,bounds,payload

    def test_author_nested_zip_streams_to_canonical_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,raw,bounds,payload=self.fixture(Path(tmp))
            extract_graphs(source,raw,bounds)
            self.assertEqual((raw/'normal/xlabel_normal.pickle').read_bytes(),payload)
            self.assertEqual(sorted(p.name for p in raw.iterdir()),['normal'])

    def test_direct_member_does_not_traverse_archive_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,raw,bounds,payload=self.fixture(root,nested=False)
            extract_graphs(source,raw,bounds)
            self.assertEqual((raw/'normal/xlabel_normal.pickle').read_bytes(),payload)
            self.assertFalse((root/'xlabel_normal.pickle').exists())

    def test_multiple_possible_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,raw,bounds,_=self.fixture(Path(tmp),ambiguous=True)
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                extract_graphs(source,raw,bounds)

    def test_missing_nested_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,raw,bounds,_=self.fixture(Path(tmp),missing=True)
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                extract_graphs(source,raw,bounds)

    def test_wrong_raw_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,raw,bounds,_=self.fixture(Path(tmp),wrong_size=True)
            with self.assertRaisesRegex(ValueError,'size'):
                extract_graphs(source,raw,bounds)

if __name__=='__main__':
    unittest.main()
