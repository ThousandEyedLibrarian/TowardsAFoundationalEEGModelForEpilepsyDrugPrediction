from eeg_analysis import AnalysisConfig, EEGAnalyzer

config = AnalysisConfig(
    input_file="sub-NDARAC904DMU_task-FunwithFractals_eeg.set",
    output_dir="./detailed_results",
    h_freq=45.0,
    notch_freq=None,
    plot_connectivity=True,  # Add connectivity plot
    plot_raw=True,  # Add raw data visualization
)

analyzer = EEGAnalyzer(config)
analyzer.run_full_analysis()
