"""Original all33 prepare guard and accepted four-field epoch binding."""
from pathlib import Path

def guarded_prepare(original, *, library_path, library_identity, error_type):
    expected_path = Path(library_path).resolve()
    expected_identity = tuple(library_identity)

    def prepare(self, *, device, capturing=False):
        try:
            # The normal runtime performs its original all-entry query and
            # generic admission. This wrapper performs no native query.
            records = original(self, device=device, capturing=capturing)
            if self.path.resolve() != expected_path or self.identity != expected_identity or self.profile is not None:
                raise error_type('A59 resource guard requires the exact ordinary library/plan.')
            entries = self.manifest['entries']
            if len(entries) != 33 or len(records) != 33 or [r['entry_id'] for r in records] != list(range(33)):
                raise error_type('A59 resource guard requires all 33 original admissions.')
            for i in (13, 28):
                row, entry = records[i], entries[i]
                if (entry['entry_id'] != i or entry['grid'] != [96, 1, 1] or
                    entry['block_threads'] != 128 or entry['dynamic_shared_bytes'] != 0 or
                    row['block_threads'] != 128 or row['resident_blocks_per_sm'] < 12 or
                    row['local_bytes'] != 0 or row['static_shared_bytes'] != 16 or
                    row['dynamic_shared_bytes'] != entry['dynamic_shared_bytes']):
                    raise error_type('A59 terminal NormStats resource class failed before compute: entry ' + str(i))
            return records
        except BaseException:
            self.invalidate()
            raise

    return prepare

def install(side):
    from triton.flagmega.errors import RuntimeContractError
    from triton.flagmega.runtime.cpp_programs import CppProgramLibrary
    artifact = side['artifact']
    contract = (artifact['library']['path'], artifact['library']['sha256'],
                artifact['source']['sha256'], side['plan_sha256'],
                side['counter_epoch_plan_sha256'])
    previous = getattr(CppProgramLibrary.prepare, '_a59_guard_contract', None)
    if previous is not None:
        if previous != contract:
            raise RuntimeContractError('Conflicting current ordinary/epoch resource guard.')
        return
    original = CppProgramLibrary.prepare
    wrapper = guarded_prepare(original, library_path=contract[0],
                              library_identity=contract[1:], error_type=RuntimeContractError)
    wrapper._a59_guard_contract = contract
    wrapper._a59_original_prepare = original
    CppProgramLibrary.prepare = wrapper
