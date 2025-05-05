import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_timedelta64_dtype, is_list_like

def print_signal_with_qrs(signal_data, qrs_predicted, true_qrs=None, mini=0, maxi=None, description=""):
    """Plots a segment of an ECG signal and marks predicted and true QRS complexes.

    This function handles both NumPy arrays and Pandas DataFrames (1 or 2 columns).
    For DataFrames with a time column, `mini` and `maxi` can be timestamps.
    QRS markers (`qrs_predicted`, `true_qrs`) should match the input type:
    indices for NumPy arrays/1-col DataFrames, timestamps for 2-col DataFrames.

    Args:
        signal_data (np.ndarray | pd.DataFrame): The ECG signal data.
            - If NumPy array: 1D array of signal values.
            - If DataFrame: 
                - 1 column: Numeric signal values.
                - 2 columns: One numeric (signal), one datetime-like (time).
        qrs_predicted (list-like | pd.Index): Indices or timestamps of predicted QRS.
        true_qrs (list-like | pd.Index, optional): Indices or timestamps of true QRS.
            Defaults to None.
        mini (int | str | pd.Timestamp | datetime, optional): Start index or timestamp
            for plotting. Defaults to 0.
        maxi (int | str | pd.Timestamp | datetime, optional): End index or timestamp
            for plotting. Defaults to end of signal.
        description (str, optional): Title for the plot. Defaults to "".
    """
    is_dataframe = isinstance(signal_data, pd.DataFrame)
    time_col = None
    signal_col = None
    time_axis = None
    signal_values = None

    # --- Input Validation and Data Extraction ---
    if is_dataframe:
        if signal_data.shape[1] == 1:
            signal_col = signal_data.columns[0]
            if not is_numeric_dtype(signal_data[signal_col]):
                raise ValueError("1-column DataFrame must be numeric.")
            signal_values = signal_data[signal_col].to_numpy()
            time_axis = signal_data.index # Use DF index as time axis
        elif signal_data.shape[1] == 2:
            for col in signal_data.columns:
                if is_numeric_dtype(signal_data[col]) and not pd.api.types.is_bool_dtype(signal_data[col]):
                    if signal_col is not None: raise ValueError("2-col DataFrame: Too many numeric columns.")
                    signal_col = col
                elif is_datetime64_any_dtype(signal_data[col]) or is_timedelta64_dtype(signal_data[col]):
                    if time_col is not None: raise ValueError("2-col DataFrame: Too many time-like columns.")
                    time_col = col
            if signal_col is None or time_col is None:
                raise ValueError("2-col DataFrame: Could not identify one signal and one time column.")
            signal_values = signal_data[signal_col].to_numpy()
            time_axis = signal_data[time_col]
        else:
            raise ValueError("Input DataFrame must have 1 or 2 columns.")
    elif isinstance(signal_data, np.ndarray):
        if signal_data.ndim != 1:
            raise ValueError("Input NumPy array must be 1-dimensional.")
        signal_values = signal_data
        time_axis = np.arange(len(signal_values)) # Use indices as time axis
    else:
        raise TypeError("signal_data must be a NumPy array or Pandas DataFrame.")

    # --- Determine Plot Range (Indices) ---
    start_idx = 0
    end_idx = len(signal_values)

    if time_col:
        # Input has timestamps, mini/maxi might be timestamps
        time_series = pd.Series(time_axis.to_numpy(), index=time_axis) # Ensure we can index by time
        try:
            if mini != 0:
                 # Attempt to convert mini to timestamp and find corresponding index
                 ts_mini = pd.Timestamp(mini)
                 # Find the first index where time >= ts_mini
                 start_idx = time_series.searchsorted(ts_mini, side='left')
            if maxi is not None:
                 ts_maxi = pd.Timestamp(maxi)
                 # Find the first index where time > ts_maxi
                 end_idx = time_series.searchsorted(ts_maxi, side='right') # Use right to include maxi
        except Exception as e:
            print(f"Warning: Could not parse mini/maxi as timestamps ({e}). Falling back to index-based plotting.")
            # Fallback to integer indices if timestamp conversion fails
            if isinstance(mini, int): start_idx = mini
            if isinstance(maxi, int): end_idx = maxi
            else: end_idx = len(signal_values)

    else:
        # Input uses indices
        if isinstance(mini, int): start_idx = mini
        if maxi is None: # Handle default maxi=None
             end_idx = len(signal_values)
        elif isinstance(maxi, int):
             end_idx = maxi
        else:
             print(f"Warning: maxi='{maxi}' is not an integer for index-based data. Plotting entire signal.")
             end_idx = len(signal_values)

    # Ensure indices are valid
    start_idx = max(0, start_idx)
    end_idx = min(len(signal_values), end_idx)
    if start_idx >= end_idx:
        print("Warning: Plot range is empty (start_idx >= end_idx). Nothing to plot.")
        return

    # --- Slice Data for Plotting ---
    signal_cut = signal_values[start_idx:end_idx]
    time_cut = time_axis[start_idx:end_idx]

    # --- Plotting ---
    plt.figure(figsize=(15, 4)) # Wider default figure
    plt.plot(time_cut, signal_cut, label='Signal')

    # --- Plot QRS Markers ---
    def plot_markers(qrs_data, label, color, marker):
        if qrs_data is None or len(qrs_data) == 0:
            return

        qrs_in_range_times_plot = [] # Variable pour les positions X (temps ou indice)
        qrs_in_range_values_plot = [] # Variable pour les positions Y (valeur du signal)

        if time_col:
             # Expect timestamps
             if not isinstance(qrs_data, (pd.Index, pd.Series)):
                  try:
                      qrs_data = pd.Index(qrs_data)
                  except Exception as convert_e:
                      print(f"Warning: Could not convert {label} QRS data to Index ({convert_e}). Skipping markers.")
                      return

             try:
                 qrs_ts = pd.to_datetime(qrs_data)
                 
                 # Utiliser .iloc pour accéder par position entière (premier/dernier élément de la slice)
                 ts_start = pd.to_datetime(time_cut.iloc[0] if len(time_cut) > 0 else time_axis.iloc[0])
                 ts_end = pd.to_datetime(time_cut.iloc[-1] if len(time_cut) > 0 else time_axis.iloc[-1])

                 mask = (qrs_ts >= ts_start) & (qrs_ts <= ts_end)
                 qrs_in_range_times = qrs_ts[mask]
                 
                 if len(qrs_in_range_times) == 0:
                     return # Nothing to plot

                 # Utilisation de l'itération pour récupérer les valeurs (solution précédente)
                 signal_df_indexed = signal_data.set_index(time_col)
                 successful_timestamps = []
                 successful_values = []
                 for ts in qrs_in_range_times:
                     try:
                         value = signal_df_indexed.loc[ts, signal_col]
                         successful_timestamps.append(ts)
                         successful_values.append(value)
                     except KeyError:
                         # Ignorer silencieusement si un timestamp QRS n'est pas exactement trouvé
                         pass 
                     except Exception as e_loc:
                          print(f"Warning: Unexpected error during .loc for {ts}: {e_loc}. Skipping point.")
                          pass
                 
                 qrs_in_range_times_plot = successful_timestamps
                 qrs_in_range_values_plot = successful_values

             except Exception as e:
                 print(f"Warning: Error processing timestamps for {label}. Skipping markers. Details: {type(e).__name__}: {e}")
                 return
        else:
            # Expect indices
            if not is_list_like(qrs_data):
                 print(f"Warning: {label} QRS data is not list-like. Skipping markers.")
                 return
            qrs_indices = np.array(qrs_data)
            mask = (qrs_indices >= start_idx) & (qrs_indices < end_idx)
            valid_indices = qrs_indices[mask]
            qrs_in_range_times_plot = valid_indices
            # Assurer que les indices sont valides avant d'accéder à signal_values
            if len(valid_indices) > 0:
                 # Vérifier les bornes au cas où
                 valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < len(signal_values))]
                 qrs_in_range_values_plot = signal_values[valid_indices]
            else:
                 qrs_in_range_values_plot = []

        # Appel final à Scatter
        if len(qrs_in_range_times_plot) > 0 and len(qrs_in_range_times_plot) == len(qrs_in_range_values_plot):
             plt.scatter(qrs_in_range_times_plot, qrs_in_range_values_plot, color=color, label=label, marker=marker, s=100, zorder=3)
        elif len(qrs_in_range_times_plot) > 0:
             print(f"Warning: Mismatch between plot times ({len(qrs_in_range_times_plot)}) and values ({len(qrs_in_range_values_plot)}). Cannot scatter plot {label}.")

    plot_markers(qrs_predicted, 'Predicted QRS', 'red', 'x')
    if true_qrs is not None:
        plot_markers(true_qrs, 'True QRS', 'green', 'o')

    # --- Final Touches ---
    if description != "":
        plt.title(label=description)

    plt.xlabel("Time" if time_col else "Sample Index")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)
    # plt.show() # Removed: Let the calling script handle plt.show()