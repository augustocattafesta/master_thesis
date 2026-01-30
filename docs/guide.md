# **Analysis and configuration guide**

## Run the analysis

After the installation, the CLI can be used with the command:
```bash
analysis config paths
```

The .yaml configuration file path and the data files or folders path are the arguments needed to run the analysis.

To analyze one or more source files, the command needs a sequence of source file paths and a calibration with pulsed data as the last argument. The .yaml configuration file must always be the first argument to be passed to the command line. The following is an example of the analysis of two source files using the same calibration file:

```bash
analysis path_config path_file0 path_file1 path_calibration
```

To analyze one or more folders, the order of the arguments is the same as the previous case, with the exception that no calibration file has to be specified, as it is supposed that each folder contains its own calibration file:

```bash
analysis path_config path_folder0 path_folder1
```

The default directory to search for the data is the data directory in the package root. If a folder is inside this directory, there is no need to specify all the path of the folder, but the relative path is enough to locate it. This works for all the files and folders, as long as the path relative to the data directory is specified. 

It is also possible to save the analysis results (e.g. plots and numerical values) inside the results directory, which is automatically created in the user home. This option can be enabled using the optional argument in the CLI command. It is also possible to choose the format to use to save the plots (pdf or png). An example of command to save the results is:

```bash
analysis path_config path_folder0 -s -f png
```

To run the analysis, a configuration file must always be specified. In the next section explains how to write a correct configuration file.

## Write the configuration file

The configuration file is a .yaml file which contains all the information about the acquisition (e.g. the date, the detector, the source, etc...) and the analysis pipeline. This file allows to write a modular pipeline, because it is possible to execute tasks and to easily configure them.

**Warning:**  when a key is optional and you want to set it to the default value, all the line must be deleted, not only the value, otherwise a type error can be raised or a `None` value can be assigned to the key.

### Acquisition

At the beginning of the configuration file, the acquisition mapping is necessary to specify the date of the acquisition, the name and type of the chip and other detector and source properties. These properties are stored as keys of the mapping, and a value is assigned to each of them. If not specified, the fields are mandatory.

```yaml
acquisition:
  date: "2026-01-01"
  chip: W1a
  structure: 86.6 um
  gas: Ar
  w: 26.        # Optional. Default is 26.0 eV
  element: Fe55
  e_peak: 5.9   # Optional. Default is 5.95 keV.
```

### Pipeline

The pipeline is the core of the analysis, where everything that has to be calculated is defined. Each step of the pipeline is a task that contains all the properties that are needed to execute it. The pipeline is structured as a sequence of tasks. Each task is a mapping, with its own key-value pairs. A task is defined by its name, and the other keys define its behaviour.

The order of tasks in the configuration file does not affect execution, as priority is automatically assigned to the `calibration` task (which is mandatory for the pipeline) followed by the `fit_spec` task, if defined.

Apart from the `calibration` and `fit_spec` tasks, all the other tasks can be written multiple times, specifying different `target` or parameters. As an example, if you have multiple emission lines in a spectrum and you want to estimate the gain from each of them, it is possible to write a `gain` task for each of them by specifying the target declared during the `fit_spec` subtasks. See the next sections for more details.

#### Calibration

This task performs the calibration between the ADC counts of the multi-channel analyzer and the charge (or voltage). It is performed using the calibration pulse files, which contain data at fixed known voltages, and after fitting each pulse with a Gaussian, a linear fit is performed to find the conversion parameters.

```yaml
pipeline:

  - task: calibration
    charge_conversion: true     # Optional, whether to convert to voltage or charge
    show: false                 # Optional, plot the calibration results
```

#### Spectral fitting

This task performs the spectral fitting on one or multiple emission lines at the same time. The task is divided into subtasks, each of them defined by the `target` (a name given to the subtask), and the `model` to use for the emission line fit (which must be *Gaussian* or *Fe55Forest*). These keys are mandatory for all the subtasks.

A subtask also has another optional key, which is `fit_pars`, that allows to specify some properties of the fitting procedure. If the key `fit_pars` is written in the configuration file, at least one property must be specified, but none of them is mandatory.

```yaml
  - task: fit_spec
    subtasks:
      - target: main_peak
        model: Fe55Forest

      - target: escape_peak
        model: Gaussian
        fit_pars:               # Optional
          xmin: 2.0             # Optional, left range limit for the fit
          xmax: 4.              # Optional, right range limit for the fit
          num_sigma_left: 1.    # Optional, number of sigma to the left of the line center for the fit
          num_sigma_right: 1.   # Optional, number of sigma to the right of the line center for the fit
          absoulute_sigma: true # Optional
          p0: [1., 1., 1.]      # Init parameters for the line fit
```

#### Gain estimate

This task performs the estimate of the gain using the spectral fitting results of a given `target`, previously specified during the *fit_spec* task.

If the analysis is performed on a single file, there are no other keys to specify. If the analysis is performed on multiple files or on one or multiple folders, the `fit`, `plot` and `label` can be specified. 

**Note:** no error is raised if these optional keys are specified during the analysis of a single file.  

```yaml
  - task: gain
    target: main_peak
    fit: true            # Optional, fit the data with an exponential model
    show: true           # Optional, plot gain vs back voltage
    label: Example label # Optional, label of the plot
    yscale: linear       # Optional, y-axis scale of the plot (linear or log)
```

#### Gain trend with time

This task performs the study of the gain as a function of time. The gain is estimated in the same way as the *gain* task. For each source file, the time and the length of the acquisition are extracted from the header. The time info are combined together to get the variation of the gain with time.

The gain trend can also be analyzed using fitting subtasks, which allow to fit the data with a model or a composition of models from *aptapy.models*. These subtasks share the same syntax of the spectral fitting subtasks.

If more than one trend is visible in the data, multiple targets can be specified, and the fit range can be adjusted to each trend.

```yaml
  - task: gain_trend
    target: main_peak
    subtasks:                         # Optional, fitting subtasks
      target: trend                   
      model: Exponential + Constant   # Composition of models 
    label: Example label              # Optional, label of the plot
```

#### Gain compare between folders

This task allows to compare the gain results of two or more different folders on the same plot. To execute this task, it is required that at least a `gain` task has been completed on a given `target` emission line.

It is also possible to combine the gain estimates from the folders and perform a single exponential fit on all the data.

```yaml
  - task: compare_gain
    target: main_peak
    combine: true                     # Optional, combine the data of all the folders 
```


#### Drift varying


#### Resolution estimate

This task performs the estimate of the resolution of the detector using the spectral fitting results of a given `target`, previously specified during the *fit_spec* task. This estimate is made using only the fitted FWHM of the line and the calibrated charge of the line center:
$$
\frac{\Delta E}{E} = \frac{\text{FWHM}}{E}.
$$
The result is reported as a percentage.

As for the gain task, the `show` and `label` keys works only if the analysis is performed on multiple files or folders

```yaml
  - task: resolution
    target: main_peak
    show: true            # Optional, plot resolution vs back voltage
    label: Example label  # Optional, label of the plot
```

#### Resolution estimate with escape peak

This task performs the estimate of the resolution of the detector using the spectral fitting results of a main line emission and the correspondent escape peak. The two emission lines are specified by the `target_main` and `target_escape` keys. The resolution is estimated as:
$$
\frac{\Delta E}{E} = \frac{\text{FWHM}}{E_{main}} \frac{E_{main} - E_{peak}}{x_{main} - x_{peak}},
$$
where $E_{main}$ and $E_{peak}$ are the theoretical emission energies of the main and the escape peaks, and $x_{main}$ and $x_{peak}$ are the fitted line center positions. This estimate and the previous lead to the same result if the charge calibration has been correctly performed.

The `show` and `fit` keys can be specified if working on multiple files.

```yaml
  - task: resolution_escape
    target_main: main_peak
    target_escape: escape_peak
```

#### Resolution compare between folders

To be implemented...

#### Plot the spectrum

This task is used to plot the spectrum of all the analyzed files. It is possible to plot the spectrum without performing the spectral fitting before writing only the name of the task.

In the case of spectral fitting and physical quantities estimate, it is possible to specify the `target` to plot, the general `label` of the plot, if you want to show the quantity estimates in the legend using `task_labels` and where to show the legend with `loc`. It is also possible to set a specific x-axis range for the plot using `xrange`.

```yaml
  - task: plot
    targets:              # Optional, target fit to show
      - main_peak
      - escape_peak
    label: Example label  # Optional, general label of the plot
    task_labels:          # Optional, estimated quantities to show in the legend
      - gain
    loc: best             # Optional, position of the legend in the plot
    xrange: [0., 3.]      # Optional, the x-axis range to show on the plot
    show: true            # Optional, whether to show the plot
```
