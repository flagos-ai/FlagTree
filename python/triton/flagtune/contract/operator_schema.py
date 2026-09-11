# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Safe parsing and compilation for FlagTune training and model configs.

Training configs may describe several variants, while an exported model config
describes exactly one model. Parsing is deliberately stateless: no process
registry, module discovery, arbitrary import, or user-provided callable is
involved. Runtime lookup is determined by the complete
``(platform_key, op_id, variant, dtype_key)`` identity and its model bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from triton.flagtune._dependencies import require_optional_dependency
from triton.flagtune.core.interfaces import ParameterField, ParameterSpace
from triton.flagtune.contract.expressions import (
    SafeExpressionError,
    evaluate_expression,
    require_mapping as _require_mapping,
    validate_expression as _validate_expression,
    validate_symbol_name,
)
from triton.flagtune.contract.identity import (
    ModelIdentity,
    make_dtype_key,
    make_platform_key,
    normalize_dtype_name,
    validate_identity_segment,
    validate_op_id,
    validate_variant_name,
)

Expression = Any
Operation = Callable[..., Any]

_MISSING = object()
_GPU_METADATA_FIELDS = frozenset({
    "backend",
    "vendor",
    "device_name",
    "architecture",
    "platform_key",
})


def _ident(value: Any) -> Any:
    return value


def _add(lhs: Any, rhs: Any) -> Any:
    return lhs + rhs


def _sub(lhs: Any, rhs: Any) -> Any:
    return lhs - rhs


def _mul(lhs: Any, rhs: Any) -> Any:
    return lhs * rhs


def _div(lhs: Any, rhs: Any) -> Any:
    return lhs // rhs


def _cdiv(lhs: Any, rhs: Any) -> Any:
    return -(-lhs // rhs)


def _fdiv(lhs: Any, rhs: Any) -> Any:
    return lhs / rhs


def _mod(lhs: Any, rhs: Any) -> Any:
    return lhs % rhs


def _pow(value: Any, exponent: Any) -> Any:
    return value**exponent


def _alignup(value: Any, alignment: Any) -> Any:
    return _cdiv(value, alignment) * alignment


def _aligndown(value: Any, alignment: Any) -> Any:
    return _div(value, alignment) * alignment


def _all(*values: Any) -> bool:
    return all(values)


def _any(*values: Any) -> bool:
    return any(values)


def _not(value: Any) -> bool:
    return not value


BUILTIN_OPS: Dict[str, Operation] = {
    "ident": _ident,
    "identity": _ident,
    "add": _add,
    "sub": _sub,
    "mul": _mul,
    "div": _div,
    "cdiv": _cdiv,
    "fdiv": _fdiv,
    "mod": _mod,
    "log2": math.log2,
    "pow": _pow,
    "alignup": _alignup,
    "aligndown": _aligndown,
    "eq": lambda lhs, rhs: lhs == rhs,
    "ne": lambda lhs, rhs: lhs != rhs,
    "lt": lambda lhs, rhs: lhs < rhs,
    "le": lambda lhs, rhs: lhs <= rhs,
    "gt": lambda lhs, rhs: lhs > rhs,
    "ge": lambda lhs, rhs: lhs >= rhs,
    "all": _all,
    "any": _any,
    "not": _not,
}
"""Built-in operations available to every FlagTune expression.

Arithmetic uses Python semantics except that ``div`` is floor division and
``cdiv`` is ceiling integer division.  ``pow(x, p)`` computes ``x ** p``.
``alignup`` and ``aligndown`` assume a non-zero positive alignment.  Custom
Model configs may use only these operations so exported bundles remain
self-contained and safe to load.
"""

FlagTuneConfigError = SafeExpressionError


@dataclass(frozen=True)
class InputSpec:
    """Describe one ordered input in a variant's shape space.

    Attributes:
        name: Variable name exposed to expressions.
        default: Optional literal or expression used when the caller omits the
            input.  Defaults can reference only earlier inputs because inputs
            are normalized in YAML declaration order.
        minimum: Optional inclusive lower-bound expression.
        maximum: Optional inclusive upper-bound expression.

    Notes:
        Missing optional fields use an internal sentinel rather than ``None``;
        therefore ``None`` remains a valid explicit expression literal.
    """

    name: str
    default: Any = _MISSING
    minimum: Any = _MISSING
    maximum: Any = _MISSING


@dataclass(frozen=True)
class FeatureSpec:
    """Bind an ordered model feature name to its FlagTune expression.

    Features are evaluated in declaration order.  Later features may refer to
    earlier named features, and the resulting order is the exact XGBoost input
    column order; no implicit dtype or kernel-kind feature is added.
    """

    name: str
    expression: Expression


@dataclass
class VariantInfo:
    """Compiled metadata and helpers for one executable operator variant.

    A variant corresponds to one kernel implementation family. It owns the shape-input
    schema, eligibility expression, tunable parameter space, ordered model
    features, and expression operation table produced at registration time.

    Attributes:
        op_id: Globally namespaced logical operator ID, such as ``"flaggems/mm"``.
        name: Variant name, such as ``"general_tma"``.
        inputs: Ordered input specifications.
        when: Eligibility expression evaluated from normalized inputs.
        param_space: Cartesian tunable-parameter space for this variant.
        features: Ordered XGBoost feature specifications.
        operations: Built-in expression operation table.
    """

    op_id: str
    name: str
    inputs: List[InputSpec]
    when: Expression
    param_space: ParameterSpace
    features: List[FeatureSpec]
    operations: Mapping[str, Operation]

    @property
    def input_names(self) -> List[str]:
        """Return shape-input names in configuration declaration order."""
        return [field.name for field in self.inputs]

    @property
    def param_names(self) -> List[str]:
        """Return all tunable parameter names in declaration order."""
        return self.param_space.all_field_names

    @property
    def feature_names(self) -> List[str]:
        """Return the exact ordered feature sequence expected by the model."""
        return [feature.name for feature in self.features]

    def normalize_inputs(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve and validate runtime shape inputs for this variant.

        Args:
            values: Mapping containing runtime arguments.  Extra keys are
                ignored; declared inputs are copied or defaulted in declaration
                order.

        Returns:
            A new dictionary containing exactly the declared input names, in
            declaration order, with defaults evaluated.

        Raises:
            FlagTuneConfigError: If a required input is missing or a value is
                outside its inclusive ``min``/``max`` bounds.
            KeyError: If a default or bound expression references a value that
                is unavailable at the point it is evaluated.

        Notes:
            Default and bound expressions can see normalized earlier inputs and
            the synthetic ``inputs`` mapping.  They cannot depend on later
            inputs, tunable parameters, or features.
        """
        normalized: Dict[str, Any] = {}
        for field in self.inputs:
            if field.name in values:
                value = values[field.name]
            elif field.default is not _MISSING:
                context = {**normalized, "inputs": normalized}
                value = evaluate_expression(field.default, context, self.operations)
            else:
                raise FlagTuneConfigError(f"missing required input {field.name!r} for {self.op_id}/{self.name}")

            context = {**normalized, field.name: value, "inputs": {**normalized, field.name: value}}
            if field.minimum is not _MISSING:
                minimum = evaluate_expression(field.minimum, context, self.operations)
                if value < minimum:
                    raise FlagTuneConfigError(f"input {field.name!r}={value!r} is below minimum {minimum!r}")
            if field.maximum is not _MISSING:
                maximum = evaluate_expression(field.maximum, context, self.operations)
                if value > maximum:
                    raise FlagTuneConfigError(f"input {field.name!r}={value!r} is above maximum {maximum!r}")
            normalized[field.name] = value
        return normalized

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Return whether normalized inputs satisfy the variant's ``when`` expression.

        Input normalization and bounds checking happen before eligibility is
        evaluated, so malformed shapes raise rather than simply returning
        ``False``.
        """
        inputs = self.normalize_inputs(values)
        context = {**inputs, "inputs": inputs}
        return bool(evaluate_expression(self.when, context, self.operations))

    def iter_configs(self) -> Iterable[Dict[str, Any]]:
        """Yield every legal parameter combination in deterministic order.

        The underlying space is a Cartesian product and is generated lazily.
        Its size therefore grows multiplicatively with each parameter field.
        """
        return self.param_space.iter_configs()

    def build_feature_rows(
        self,
        values: Mapping[str, Any],
        configs: Iterable[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build named, ordered feature rows for candidate configurations.

        Args:
            values: Runtime shape values accepted by :meth:`normalize_inputs`.
            configs: Candidate parameter mappings.  Every declared parameter
                must be present; extra keys are ignored.

        Returns:
            One insertion-ordered feature dictionary per candidate.  Each
            expression can reference normalized inputs, parameters, the
            ``inputs`` and ``params`` mappings, and earlier named features.

        Raises:
            KeyError: If a candidate omits a declared parameter or an
                expression references unavailable data.
            FlagTuneConfigError: If runtime inputs cannot be normalized.

        Notes:
            No ``kernel_kind_code`` or ``dtype_code`` is injected. Tensor dtype
            combinations are isolated by ``dtype_key`` outside this feature matrix.
        """
        inputs = self.normalize_inputs(values)
        rows: List[Dict[str, Any]] = []
        for raw_config in configs:
            params = {name: raw_config[name] for name in self.param_names}
            context: Dict[str, Any] = {**inputs, **params, "inputs": inputs, "params": params}
            row: Dict[str, Any] = {}
            for feature in self.features:
                value = evaluate_expression(feature.expression, context, self.operations)
                row[feature.name] = value
                context[feature.name] = value
            rows.append(row)
        return rows

    def build_feature_matrix(
        self,
        values: Mapping[str, Any],
        configs: Iterable[Mapping[str, Any]],
    ) -> Any:
        """Build the two-dimensional floating-point model input matrix.

        Rows correspond to ``configs`` and columns follow
        :attr:`feature_names` exactly.  Values must be convertible to NumPy
        ``float``; categorical encoding is not performed implicitly.  An empty
        candidate iterable produces an empty one-dimensional NumPy array, so
        normal prediction paths avoid calling this method with no candidates.
        """
        np = require_optional_dependency(
            "numpy",
            distribution_name="numpy",
            feature="FlagTune feature-matrix construction",
        )

        rows = self.build_feature_rows(values, configs)
        return np.asarray([[row[name] for name in self.feature_names] for row in rows], dtype=float)

    def to_config(self, config: Mapping[str, Any]) -> Any:
        """Convert a parameter mapping to :class:`triton.Config`.

        ``num_warps``, ``num_stages``, and ``num_ctas`` are removed from kernel
        keyword arguments and converted to Triton launch metadata.  Missing
        launch fields default to 4, 3, and 1 respectively.  All remaining
        entries are preserved as kernel constexpr arguments.

        Returns:
            A newly allocated ``triton.Config`` without a pre-hook.

        Raises:
            TypeError: If a launch field cannot be converted to ``int`` or the
                installed Triton ``Config`` API is incompatible.
        """
        from triton import Config

        kwargs = dict(config)
        num_warps = int(kwargs.pop("num_warps", 4))
        num_stages = int(kwargs.pop("num_stages", 3))
        num_ctas = int(kwargs.pop("num_ctas", 1))
        return Config(kwargs=kwargs, num_warps=num_warps, num_stages=num_stages, num_ctas=num_ctas)


@dataclass
class OperatorInfo:
    """Compiled definition of one globally identified training operator.

    Attributes:
        op_id: Globally namespaced operator identity used for model provenance.
        variants: Mapping from variant names to compiled metadata. The object is
            returned to the caller and never installed in process-global state.
    """

    op_id: str
    variants: Dict[str, VariantInfo]

    def get_variant(self, name: str) -> VariantInfo:
        """Return one variant or raise an informative ``KeyError``.

        The error lists currently registered variants to make misspellings and
        stale variant names easier to diagnose.
        """
        try:
            return self.variants[name]
        except KeyError as exc:
            raise KeyError(f"Unknown variant {name!r} for operator {self.op_id!r}. "
                           f"Registered variants: {sorted(self.variants)}") from exc


def _parse_inputs(
    raw_inputs: Any,
    operations: Mapping[str, Operation],
    location: str,
) -> List[InputSpec]:
    mapping = _require_mapping(raw_inputs, location)
    variables = {"inputs"}
    result: List[InputSpec] = []
    for raw_name, raw_spec in mapping.items():
        name = validate_symbol_name(raw_name, f"{location} key")
        spec = _require_mapping(raw_spec, f"{location}.{name}")
        unknown = set(spec) - {"default", "min", "max"}
        if unknown:
            raise FlagTuneConfigError(f"{location}.{name} has unknown keys: {sorted(unknown)}")
        for key in ("default", "min", "max"):
            if key in spec:
                _validate_expression(spec[key], operations, variables, f"{location}.{name}.{key}")
        result.append(
            InputSpec(
                name=name,
                default=spec.get("default", _MISSING),
                minimum=spec.get("min", _MISSING),
                maximum=spec.get("max", _MISSING),
            ))
        variables.add(name)
    return result


def _parse_params(raw_params: Any, location: str) -> ParameterSpace:
    mapping = _require_mapping(raw_params, location)
    fields: List[ParameterField] = []
    for raw_name, raw_spec in mapping.items():
        name = validate_symbol_name(raw_name, f"{location} key")
        spec = _require_mapping(raw_spec, f"{location}.{name}")
        unknown = set(spec) - {"values"}
        if unknown:
            raise FlagTuneConfigError(f"{location}.{name} has unknown keys: {sorted(unknown)}")
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise FlagTuneConfigError(f"{location}.{name}.values must be a non-empty list")
        fields.append(ParameterField(name=name, legal_values=list(values)))
    if not fields:
        raise FlagTuneConfigError(f"{location} must define at least one tunable parameter")
    return ParameterSpace(fields=fields)


def _parse_features(
    raw_features: Any,
    operations: Mapping[str, Operation],
    base_variables: set[str],
    location: str,
) -> List[FeatureSpec]:
    if not isinstance(raw_features, list) or not raw_features:
        raise FlagTuneConfigError(f"{location} must be a non-empty list")
    variables = set(base_variables)
    result: List[FeatureSpec] = []
    for index, raw_feature in enumerate(raw_features):
        item_location = f"{location}[{index}]"
        if isinstance(raw_feature, str):
            name = raw_feature
            expression = raw_feature
        elif isinstance(raw_feature, Mapping):
            name = validate_symbol_name(raw_feature.get("name"), f"{item_location}.name")
            expression = {key: value for key, value in raw_feature.items() if key != "name"}
        else:
            raise FlagTuneConfigError(f"{item_location} must be a variable or named expression")
        if name in {feature.name for feature in result}:
            raise FlagTuneConfigError(f"{item_location} duplicates feature name {name!r}")
        _validate_expression(expression, operations, variables, item_location)
        result.append(FeatureSpec(name=name, expression=expression))
        variables.add(name)
    return result


def parse_operator_config(config: Mapping[str, Any]) -> OperatorInfo:
    """Validate and compile one multi-variant training configuration.

    Args:
        config: Mapping parsed from a FlagTune training config. It must contain an
            globally namespaced ``op_id`` and a non-empty ``variants`` mapping.
            Each variant defines ordered ``inputs``, optional ``when``, non-empty
            ``params``, and ordered named ``features``.

    Returns:
        A new :class:`OperatorInfo` containing compiled variants. No global
        registry or other process state is changed.

    Raises:
        FlagTuneConfigError: If schema validation fails, an expression refers
            to an unknown variable/operation, or identity fields are unsafe.

    Notes:
        Only :data:`BUILTIN_OPS` are accepted. Feature order is preserved as
        the model column order. Model paths are derived later from ``op_id`` and
        the single-segment variant name.

    Example:
        ``parse_operator_config({"op_id": "flaggems/mm", "variants": {...}})``
    """
    root = _require_mapping(config, "config")
    op_id = validate_op_id(root.get("op_id"), "config.op_id")
    unknown_root = set(root) - {"op_id", "variants"}
    if unknown_root:
        raise FlagTuneConfigError(f"config has unknown keys: {sorted(unknown_root)}")
    operations: Dict[str, Operation] = dict(BUILTIN_OPS)

    raw_variants = _require_mapping(root.get("variants"), "config.variants")
    if not raw_variants:
        raise FlagTuneConfigError("config.variants must not be empty")

    variants: Dict[str, VariantInfo] = {}
    for raw_variant_name, raw_variant in raw_variants.items():
        variant_name = validate_variant_name(raw_variant_name, "config.variants key")
        location = f"config.variants.{variant_name}"
        spec = _require_mapping(raw_variant, location)
        unknown = set(spec) - {"inputs", "when", "params", "features"}
        if unknown:
            raise FlagTuneConfigError(f"{location} has unknown keys: {sorted(unknown)}")

        inputs = _parse_inputs(spec.get("inputs"), operations, f"{location}.inputs")
        param_space = _parse_params(spec.get("params"), f"{location}.params")
        variables = {field.name for field in inputs} | {"inputs"}
        when = spec.get("when", True)
        _validate_expression(when, operations, variables, f"{location}.when")
        feature_variables = variables | set(param_space.all_field_names) | {"params"}
        features = _parse_features(spec.get("features"), operations, feature_variables, f"{location}.features")

        variants[variant_name] = VariantInfo(
            op_id=op_id,
            name=variant_name,
            inputs=inputs,
            when=when,
            param_space=param_space,
            features=features,
            operations=operations,
        )

    return OperatorInfo(op_id=op_id, variants=variants)


def load_operator_config(path: str | Path) -> OperatorInfo:
    """Safely load and compile a multi-variant training YAML file.

    Args:
        path: Local YAML path. Relative paths are resolved against the current
            working directory, not the importing module.

    Returns:
        A stateless compiled :class:`OperatorInfo`.

    Raises:
        ImportError: If optional dependency PyYAML is unavailable.
        OSError: If the file cannot be opened.
        yaml.YAMLError: If YAML parsing fails.
        FlagTuneConfigError: If the parsed document is not a valid config.

    Notes:
        ``yaml.safe_load`` is used. A schema-version-3 integration document may
        additionally contain a ``pretune`` section; FlagTree deliberately
        ignores that FlagGems-owned section after rejecting all other root
        keys. This function adds no registration, custom-callable, path-search,
        or auto-discovery behavior.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("FlagTune YAML config loading requires PyYAML") from exc

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise FlagTuneConfigError(f"invalid FlagTune YAML in {config_path}: {exc}") from exc
    if isinstance(config, Mapping) and "schema_version" in config:
        unknown = set(config) - {"schema_version", "op_id", "variants", "pretune"}
        if unknown:
            raise FlagTuneConfigError(f"config has unknown keys: {sorted(unknown)}")
        if config.get("schema_version") != 3:
            raise FlagTuneConfigError("config.schema_version must be 3")
        config = {"op_id": config.get("op_id"), "variants": config.get("variants")}
    return parse_operator_config(config)


def variant_to_model_config(
    variant: VariantInfo,
    identity: ModelIdentity,
    dtypes: Sequence[Any],
    gpu: Mapping[str, Any],
    model_version: str,
) -> Dict[str, Any]:
    """Serialize one compiled variant as a self-contained model config.

    The returned insertion-ordered mapping is suitable for ``yaml.safe_dump``.
    It contains exactly one model and preserves input, parameter, and feature
    declaration order. Artifact-version fields are added later by the exporter.
    """
    inputs: Dict[str, Dict[str, Any]] = {}
    for field in variant.inputs:
        spec: Dict[str, Any] = {}
        if field.default is not _MISSING:
            spec["default"] = field.default
        if field.minimum is not _MISSING:
            spec["min"] = field.minimum
        if field.maximum is not _MISSING:
            spec["max"] = field.maximum
        inputs[field.name] = spec
    params = {field.name: {"values": list(field.legal_values)} for field in variant.param_space.fields}
    features: List[Any] = []
    for feature in variant.features:
        if feature.expression == feature.name:
            features.append(feature.name)
        elif isinstance(feature.expression, Mapping):
            features.append({"name": feature.name, **dict(feature.expression)})
        else:
            features.append({"name": feature.name, "op": "ident", "args": [feature.expression]})
    if (variant.op_id, variant.name) != (identity.op_id, identity.variant):
        raise FlagTuneConfigError("variant and model identity operator pair do not match")
    canonical_dtypes = [normalize_dtype_name(value) for value in dtypes]
    if make_dtype_key(canonical_dtypes) != identity.dtype_key:
        raise FlagTuneConfigError("model identity dtype_key does not match ordered dtypes")
    unknown_gpu_fields = set(gpu) - _GPU_METADATA_FIELDS
    missing_gpu_fields = _GPU_METADATA_FIELDS - set(gpu)
    if unknown_gpu_fields:
        raise FlagTuneConfigError(f"GPU metadata has unknown keys: {sorted(unknown_gpu_fields)}")
    if missing_gpu_fields:
        raise FlagTuneConfigError(f"GPU metadata is missing keys: {sorted(missing_gpu_fields)}")
    try:
        backend = validate_identity_segment(gpu["backend"], "GPU backend")
        declared_platform_key = make_platform_key(
            str(gpu["vendor"]),
            str(gpu["device_name"]),
        )
        validate_identity_segment(gpu["architecture"], "GPU architecture")
    except (KeyError, TypeError, ValueError) as exc:
        raise FlagTuneConfigError(f"invalid GPU metadata: {exc}") from exc
    if declared_platform_key != identity.platform_key or gpu["platform_key"] != identity.platform_key:
        raise FlagTuneConfigError("model identity platform_key does not match GPU metadata")
    if backend not in ("cuda", "hip", "maca", "musa"):
        raise FlagTuneConfigError(f"unsupported GPU backend: {backend!r}")
    from triton.flagtune.contract.archive import validate_model_version

    return {
        "format_version": 5,
        "model_version": validate_model_version(model_version),
        "platform_key": identity.platform_key,
        "op_id": variant.op_id,
        "variant": variant.name,
        "dtype_key": identity.dtype_key,
        "dtypes": canonical_dtypes,
        "gpu": dict(gpu),
        "inputs": inputs,
        "when": variant.when,
        "params": params,
        "features": features,
    }


def parse_model_config(config: Mapping[str, Any]) -> VariantInfo:
    """Validate and compile one exported self-contained model config.

    Args:
        config: Mapping loaded from ``flagtune_config.yaml``.

    Returns:
        A compiled :class:`VariantInfo` used directly for candidate enumeration,
        input normalization, and ordered feature construction.

    Raises:
        FlagTuneConfigError: If identity fields, expressions, parameter space,
            feature declarations, or the format version are invalid.

    Notes:
        Compatibility fields are accepted here but enforced by the model manager,
        which knows the installed FlagTune version.
    """
    root = _require_mapping(config, "model config")
    allowed = {
        "format_version",
        "model_version",
        "flagtune_version_min",
        "flagtune_version_max",
        "platform_key",
        "op_id",
        "variant",
        "dtype_key",
        "dtypes",
        "gpu",
        "inputs",
        "when",
        "params",
        "features",
    }
    unknown = set(root) - allowed
    if unknown:
        raise FlagTuneConfigError(f"model config has unknown keys: {sorted(unknown)}")
    if type(root.get("format_version")) is not int or root["format_version"] != 5:
        raise FlagTuneConfigError("model config.format_version must be 5")
    from triton.flagtune.contract.archive import validate_model_version

    try:
        validate_model_version(root.get("model_version"))
    except ValueError as exc:
        raise FlagTuneConfigError(f"model config.model_version is invalid: {exc}") from exc
    model_identity_from_config(root)
    op_id = validate_op_id(root.get("op_id"), "model config.op_id")
    variant_name = validate_variant_name(root.get("variant"), "model config.variant")
    training_shape = {
        "op_id": op_id,
        "variants": {
            variant_name: {
                "inputs": root.get("inputs"),
                "when": root.get("when", True),
                "params": root.get("params"),
                "features": root.get("features"),
            }
        },
    }
    return parse_operator_config(training_shape).get_variant(variant_name)


def model_identity_from_config(config: Mapping[str, Any]) -> ModelIdentity:
    """Validate and return the complete GPU/operator/variant/dtype identity."""
    root = _require_mapping(config, "model config")
    identity = ModelIdentity(
        root.get("platform_key"),
        root.get("op_id"),
        root.get("variant"),
        root.get("dtype_key"),
    )
    raw_dtypes = root.get("dtypes")
    if not isinstance(raw_dtypes, list) or not raw_dtypes:
        raise FlagTuneConfigError("model config.dtypes must be a non-empty list")
    if make_dtype_key(raw_dtypes) != identity.dtype_key:
        raise FlagTuneConfigError("model config.dtype_key does not match dtypes")
    gpu = _require_mapping(root.get("gpu"), "model config.gpu")
    unknown_gpu_fields = set(gpu) - _GPU_METADATA_FIELDS
    missing_gpu_fields = _GPU_METADATA_FIELDS - set(gpu)
    if unknown_gpu_fields:
        raise FlagTuneConfigError(f"model config.gpu has unknown keys: {sorted(unknown_gpu_fields)}")
    if missing_gpu_fields:
        raise FlagTuneConfigError(f"model config.gpu is missing keys: {sorted(missing_gpu_fields)}")
    try:
        backend = validate_identity_segment(gpu["backend"], "model config.gpu.backend")
        validate_identity_segment(gpu["architecture"], "model config.gpu.architecture")
        actual_platform_key = make_platform_key(str(gpu["vendor"]), str(gpu["device_name"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise FlagTuneConfigError(f"invalid model config.gpu: {exc}") from exc
    if backend not in ("cuda", "hip", "maca", "musa"):
        raise FlagTuneConfigError(f"model config.gpu.backend is unsupported: {backend!r}")
    if actual_platform_key != identity.platform_key or gpu.get("platform_key") != identity.platform_key:
        raise FlagTuneConfigError("model config GPU metadata does not match platform_key")
    return identity


def load_model_config(path: str | Path) -> tuple[VariantInfo, Dict[str, Any]]:
    """Safely load one exported model YAML and return compiled and raw forms.

    Returning both forms lets the model manager validate version fields and the
    canonical config digest without reading or parsing the file twice.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("FlagTune model loading requires PyYAML") from exc
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise FlagTuneConfigError(f"invalid FlagTune model YAML in {config_path}: {exc}") from exc
    root = dict(_require_mapping(config, "model config"))
    return parse_model_config(root), root


def load_model_config_bytes(payload: bytes, *,
                            source: str = "flagtune_config.yaml") -> tuple[VariantInfo, Dict[str, Any]]:
    """Safely compile a UTF-8 model YAML payload read from an archive member."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("FlagTune model loading requires PyYAML") from exc
    try:
        config = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FlagTuneConfigError(f"invalid FlagTune model YAML in {source}: {exc}") from exc
    root = dict(_require_mapping(config, "model config"))
    return parse_model_config(root), root


def model_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest binding config and XGBoost files.

    Mapping keys are sorted recursively by JSON serialization while list order
    remains significant, making ordered features part of the artifact identity.
    Values must be JSON-compatible YAML scalars and collections.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
