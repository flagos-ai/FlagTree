"""Consume K231 once-reviewed actual artifacts; resources remain a runtime gate."""
import json

def verify_current(native, side, programs, checked, require):
    require(native['schema']=='flagmega.K231-actual-native-binding/v1' and native['passed'] is True,
            'Current independent K231 review required')
    require(native['status']=='native_source_and_host_metadata_qualified_device_gates_pending', 'Wrong native evidence stage')
    actual=side['artifact'];expected=json.loads(checked(side['expected_normal_descriptor']))
    require(native['artifact_root']==actual['root'] and native['actual_artifact']==dict(
        manifest=actual['manifest'],generated_source=actual['source'],library=actual['library']), 'Current native/artifact pins differ')
    require(native['expected_descriptor']==side['expected_normal_descriptor'] and programs==expected['cpp_programs'], 'Wrong complete current ordinary descriptor')
    require(native['integrated_source_freeze']==side['source_freeze'] and native['module_semantic_hash']==expected['module_semantic_hash'], 'Wrong integrated source/module')
    require(native['ordinary_program_plan_sha256']==side['plan_sha256']==expected['cpp_programs']['plan_sha256'], 'Wrong ordinary plan binding')
    require(native['counter_epoch_plan_sha256']==side['counter_epoch_plan_sha256']==expected['cpp_counter_epoch_plan']['plan_sha256'], 'Wrong epoch plan binding')
    require((native['entries'],native['invocations'],native['ordinary_scratch_reset_count'],native['actual_epoch_reset_count'])==
            (33,869,275,side['reset_count']), 'Native ordinary/epoch census differs')
    require(native['actual_reset_table']==[[r['ordinal'],r['nbytes']] for r in expected['cpp_counter_epoch_plan']['resets']],
            'Complete native reset table differs')
    require(native['original_feedback_nodes_additional'] is True, 'Full feedback boundary missing')
    require([r['entry_id'] for r in native['entry_checks']]==list(range(33)), 'Incomplete actual entry checks')
    required_checks=('all_33_signatures_device_definitions_geometry_launch_kind_dynamic_shared_match',
        'all_869_invocations_arguments_order_match','all_869_cpp_epoch_rows_and_complete_reset_table_match',
        'ordinary_program_manifest_unchanged','normal_frozen_generated_source_exact',
        'native_manifest_plan_source_execution_bindings_match','current_compile_paths_and_import_provenance_match','same_device_region_both_sides')
    require(all(native['checks'].get(k) is True for k in required_checks), 'Incomplete current native checks')
    for name in ('flagmega_program_manifest_v3','flagmega_counter_epoch_manifest_v1'):
        row=native['embedded_manifests'][name]
        require(all(row[k] is True for k in ('matches_artifact','matches_expected','matches_generated_source_literal')), 'Native immutable manifest mismatch')
    require(set(native['exports'])=={'flagmega_program_count_v3','flagmega_program_manifest_v3','flagmega_program_resource_info_v3',
        'flagmega_launch_program_v3','flagmega_counter_epoch_manifest_v1','flagmega_launch_counter_epoch_plan_v1'}, 'Current native exports incomplete')
    require(native['actual_resource_API_queried'] is False and native['GPU'] is False, 'Offline proof cannot replace actual resources')
    require(native['static_resources']==side['static_resources'],'Wrong actual static resources pin')
    resources=json.loads(checked(native['static_resources']))
    require(resources['library']==actual['library'] and set(resources['functions'])=={e['symbol'] for e in programs['entries']}, 'Incomplete current static resource table')
