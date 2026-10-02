"""
Evaluation Module
=================
Comprehensive evaluation tools for PV forecasting with scientific rigor.

Key Features:
- Daytime masking for honest evaluation
- Skill Score relative to Persistence
- Multi-horizon degradation analysis
- Weather regime performance breakdown
"""

from .metrics_factory import (
    MetricsResult,
    MetricsCalculator,
    MetricsFactory,
    DaylightMaskCalculator,
    MultiHorizonMetrics,
    ModelComparison,
    create_metrics_calculator,
    load_metrics_config,
    evaluate_predictions,
    compute_skill_score
)

from .error_analysis import (
    ErrorAnalysisResult,
    ErrorDistributionAnalyzer,
    HorizonDegradationAnalyzer,
    WeatherRegimeErrorAnalysis,
    TemporalErrorAnalysis,
    ResidualDiagnostics,
    generate_error_report
)

# Statistical Analysis Module (NEW - for thesis)
try:
    from .statistical_analysis import (
        DieboldMarianoTest,
        ResidualAnalyzer,
        FeatureImportanceAnalyzer,
        TrainingHealthAnalyzer,
        run_full_statistical_analysis
    )
    _STATS_AVAILABLE = True
except ImportError:
    _STATS_AVAILABLE = False

# Alias for convenience
HorizonErrorAnalyzer = HorizonDegradationAnalyzer

__all__ = [
    # Metrics
    'MetricsResult',
    'MetricsCalculator',
    'MetricsFactory',
    'DaylightMaskCalculator',
    'MultiHorizonMetrics',
    'ModelComparison',
    'create_metrics_calculator',
    'load_metrics_config',
    'evaluate_predictions',
    'compute_skill_score',
    # Error Analysis
    'ErrorAnalysisResult',
    'ErrorDistributionAnalyzer',
    'HorizonDegradationAnalyzer',
    'HorizonErrorAnalyzer',  # Alias
    'WeatherRegimeErrorAnalysis',
    'TemporalErrorAnalysis',
    'ResidualDiagnostics',
    'generate_error_report',
    # Statistical Analysis (NEW)
    'DieboldMarianoTest',
    'ResidualAnalyzer',
    'FeatureImportanceAnalyzer',
    'TrainingHealthAnalyzer',
    'run_full_statistical_analysis',
]
