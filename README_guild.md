# SAVER and Guild AI

This project supports the following functionality with Guild AI by way
of [guild.yml](guild.yml):

- Automate operations
- Track and manage runs
    - Capture each run as a separate experiment
    - List and filter runs
    - Inspect run files and output
    - Delete, backup and restore runs
- Compare run performance
- Diff runs
- View runs in TensorBoard

## Overview

[Guild AI](https://guild.ai) is an open source command line tool that
automates project tasks. Guild AI works by reading configuration in
[guild.yml](guild.yml) - it does not require changes to project source
code. Guild AI is similar to tools like Maven or Grunt but with
features supporting machine learning workflow.

Below is a summary of Guild AI commands that can be used with this
project.

**`guild help`** <br> Show project help including models, operation,
and supported flags.

**`guild run [MODEL]:OPERATION [FLAG=VAL]...`** <br> Runs a model
operation. Runs are tracked as unique file system artifacts that can
be managed, inspected, and compared with other runs. Flags may be
specified to change operation behavior.

**`guild runs`** <br> List runs, including run ID, model and
operation, start time, status, and label.

**`guild runs rm RUN`** <br> Delete a run where `RUN` is a run ID or
listing index. You can delete multiple runs matching various criteria.

**`guild compare`** <br> Compare run results including loss and
validation accuracy.

**`guild tensorboard`** <br> View project runs in TensorBoard. You can
view all runs or runs matching various criteria.

**`guild diff RUN1 RUN2`** <br> Diff two runs. You can diff flags,
output, dependencies, and files using a variety of diff tools.

**`guild view`** <br> Open a web based run visualizer to compare and
inspect runs.

For a complete list of commands, run:

```
$ guild --help
```

For help with a specific command, run:

```
$ guild COMMAND --help
```

## Get started

The `guild` program is part of [Guild
AI](https://github.com/guildai/guildai) and can be installed using
pip.

Follow the steps below to install Guild AI and initialize a project
environment.

### Install Guild AI

```
$ pip install guildai --upgrade
```

### Clone repository

```
$ git clone https://github.com/gar1t/SAVER.git
```

### Initialize environment

Change to the project directory:

```
$ cd SAVER
```

Initialize an environment:

```
$ guild init
```

The `init` command creates a virtual environment in `env` and installs
Guild AI and the Python packages listed in
[`requirements.txt`](requirements.txt). Environments are used to
isolate project work from other areas of the system.

Activate the environment:

```
$ source guild-env
```

Check the environment:

```
$ guild check
```

If you get errors, run `guild check --verbose` to get more information
and, if you can't resolve the issue, [open an
issue](https://github.com/guildai/guildai/issues) to get help.

## Train SAVER

To train SAVER with the default hyperparameters, run:

```
$ guild run train
```

To start training, press `Enter`.

You can alternatively use different hyperparameters, which are called
*flags*. To view available flags, run:

```
$ guild run train --help-op
```

## View training progress in TensorBoard

To view training progress in TensorBoard, open a separate command
console.

In the new command console, change to the project directory:

```
$ cd SAVER
```

Activate the environment:

```
$ source guild-env
```

List project runs:

```
$ guild runs
```

Guild shows the running `train` operation (run ID and dates will
differ):

```
[1:424ecdbc]  ./saver:train  2018-12-07 16:44:22  completed
```

Open TensorBoard:

```
$ guild tensorboard
```

If you run `guild tensorboard` on your workstation, Guild starts
TensorBoard on an available port and opens it in your browser. If you
run the command on a remote server, you have to open TensorBoard in
your browser manually. Use the link displayed in the console.

If you need to run TensorBoard on a specific port, use the `--port`
option:

```
$ guild tensorboard --port 8080
```

Guild automatically synchronizes TensorBoard with the current list of
run. You can leave TensorBoard running during your work.

## Compare model performance

You may compare model performance using TensorFlow (see steps above
for starting TensorFlow with Guild) or using the Guild AI `compare`
command.

To compare model loss and validation accuracy, run:

```
$ guild compare
```

Use the arrow keys to navigate within the Compare program.

Press `q` to exit the Compare program.

## TODO

- Verify scalar keys for loss/loss_step
