# **Examples of Analysis**

To run the analysis, first you need to write the configuration file following the [Analysis and Configuration Guide](guide.md).

## Single file analysis

In this example, a basic analysis of a single source file is performed. See the configuration YAML file snippet for more details.

```yaml
--8<-- "docs/examples/single_example_config.yaml"
```

After writing the configuration file, the analysis can be launched from the command line interface with:

```bash
analysis path_source path_calibration path_config
```

After the analysis is completed, the result is the following plot, showing the main emission line of the spectrum along with the fit model and the legend showing the gain and energy resolution results:

![Fit](figures/single_example.png)

## Folder analysis

In this example, the analysis of a folder is performed. You can see from the configuration YAML file snippet below that the configuration file doesn't differ much from that for the single file analysis. Indeed, the configuration files are quite flexible and most of the tasks can be executed both on single and multiple files or folders. The main difference for some tasks is the possibility to specify particular keys that work only on multiple files, otherwise nothing is done.

```yaml
--8<-- "docs/examples/folder_example_config.yaml"
```

The command to run is simply:
```bash
analysis path_folder path_config
```

Setting the `plot` key to `true` for the `gain` and `resolution` tasks, the output plots are the following:

![Fit](figures/folder_gain.png)
![Fit](figures/folder_resolution.png)

## Analysis of gain trend with time