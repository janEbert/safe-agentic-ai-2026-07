# Setting up a secure agentic harness

In this exercise, we're going to set up an agentic harness with some restrictions on what it and the agent can do to better protect our system and data.

We'll be running the [Pi](https://pi.dev/) agent harness in the [Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime).

## Exercises

Using the documentation below, set up `pi` running sandboxed by `srt` with access to your self-hosted model from the previous exercise.

1. Try around with `srt` and its configuration.
   1. Which commands can you execute?
   2. Are you able to read a file?
   3. Are you able to write to a file?
   4. Are you able to connect to an internet address?
   5. Now apply configuration to achieve the opposite behavior for points 2.–4.
2. Run `pi` sandboxed by `srt`.
   1. Create a sandbox configuration that allows you to run `pi`, testing to making sure that the boundaries/constraints are properly enforced (prefix a message with `!` to send it as a shell command).  
      Note, on Windows you'd run like `.\srt -- pi`.

      If you get errors indicating that `pi` cannot be found, think carefully about whether you configured the sandbox to be able to read the location where the `pi` command resides. Remember that the agent can only use executables that it can read; therefore you can use `denyRead` permissions to also hide commands from the agent.
   2. Also make sure that inside `pi`, you can configure its behavior. For example, try to set a different theme with `/settings` and make sure it stays configured when you restart `pi`.  
      This requires you to make your `PI_CODING_AGENT_DIR`, by default `~/.pi/agent`, writable.
3. Add your self-hosted model endpoint to `pi`.
   1. Set up the `models.json` in `PI_CODING_AGENT_DIR` (by default `~/.pi/agent/models.json`) (again, refer to the documentation for "Custom providers" below for help).  
      Don't forget to append the `/v1` suffix to the end of the base URL!
   2. Try to switch to your provider inside `pi` with the `/model` command.
   3. Test whether the model responds to your prompts.  
      If you get "400 Bad Request" errors, try to set the provider's `api` to `openai-completions`.

Remember that you can use the `srt` tool for _anything_! Whenever you need to run a program in a lightweight sandbox in the future, you have it as an option.

## Sandbox

The [Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime) (`srt`) is a lightweight tool to constrain file system and network access for arbitrary programs in an easy-to-configure fashion. Because Sandbox Runtime tries to use common APIs, ideally provided by the operating system, we do not need to set up a container and can run the programs on our host system.

While this solution is not as clearly separated as a container-based (or better yet, a virtual machine-based) solution, we are able to make use of a simple API and program written in the same language as our harness. Finally, we can easily apply the Sandbox Runtime to any other tooling in the future.

### Usage

Just prepend any command with `srt` to sandbox it, optionally using `--` to clearly separate the command from `srt`'s options.

```shell
srt --help
```

```shell
srt echo 'This is a test'
# Should work because files are readable by default.
```

```shell
srt -- cat test.txt
# Assuming the file exists, this should work because files are readable by
# default.
```

```shell
srt -- touch test.txt
# Should fail because files aren't writable by default.
```

```shell
srt -- curl https://example.com
# Should fail because network access isn't allowed by default.
# Same goes for, local addresses like `localhost` as well.
```

You can activate debugging information by supplying the `--debug` flag to `srt`, or specify a configuration file with the `--settings` argument:

```shell
srt --debug --settings ./sandbox.json -- curl https://example.com
```

#### Windows

If you're having trouble with permission errors, make sure that the output of `npm ls --depth 0 --parseable -g @anthropic-ai/sandbox-runtime` is _not_ in your user directory, but e.g., in a system-wide-accessible location such as `C:\Program Files`. The sandbox runs under a different user and that user probably does not have access to your user directory by default.

### Configuration

The Sandbox Runtime uses JSON for configuration, with the configuration being loaded from `~/.srt-settings.json` by default.

With an empty configuration like the following, we'd use the default settings of having everything readable, nothing writable, and nothing accessible via network:

```json
{
  "filesystem": {
    "denyRead": [],
    "allowRead": [],
    "allowWrite": [],
    "denyWrite": []
  },
  "network": {
    "allowedDomains": [],
    "deniedDomains": [],
    "allowLocalBinding": false
  }
}
```

#### File system

The sandbox allows us to specify which files should be readable and writable. By default, all files are readable and no files are writable. We can then apply broad rules to deny reads or allow writes using `filesystem.denyRead` and `filesystem.allowWrite`. We can then selectively override those broad rules to narrowly allow reads or deny writes again using `filesystem.allowRead` and `filesystem.denyWrite`.

The priority/precedence order of these flags is the reverse order they are mentioned here. More specific flags overwrite broader ones. As a diagram, where `X > Y` indicates that `X` has higher priority than `Y`:

```
allowRead > denyRead > reading allowed by default

denyWrite > allowWrite > writing denied by default
```

If we wanted to
- disallow reading and writing everywhere
- except for allowing reading of system paths and
- except for allowing reading and writing in the directory in which we started `srt`,
we could achieve it like this, where we still need to replace `<srt-installation-path>`:

```json
{
  "filesystem": {
    "denyRead": ["~"],
    "allowRead": ["<srt-installation-path>"],
    "allowWrite": ["."],
    "denyWrite": []
  },
}
```

We do need to allow the installation directory of `srt` as well, so it can read its own required files. You can find the path to insert in place of `<srt-installation-path>` with the following command:

```shell
npm ls --depth 0 --parseable -g @anthropic-ai/sandbox-runtime
```

#### Network

Similar to the file system writing constraints, network access is denied by default, can be broadly enabled using `network.allowedDomains`, and selectively disabled using `network.deniedDomains`.

Again, priority/precedence order is higher for more narrow ones. As a diagram, where `X > Y` indicates that `X` has higher priority than `Y`:

```
deniedDomains > allowedDomains > network access denied default
```

If we wanted to
- disallow network access to everywhere
- except for Anthropic, OpenAI, and NVIDIA inference endpoints,
we could achieve it like this:

```json
{
  "network": {
    "allowedDomains": [
      "api.anthropic.com"
      "claude.ai",
      "platform.claude.com",

      "auth.openai.com",
      "api.openai.com",
      "chatgpt.com",

      "integrate.api.nvidia.com"
    ],
    "deniedDomains": [],
    "allowLocalBinding": false
  }
}
```

## Harness

[Pi](https://pi.dev/) is an agentic harness designed with efficiency, minimalism, and extensibility in mind. It's also the harness behind [OpenClaw](https://openclaw.ai/).

Pi already has a lot of existing model providers pre-configured, so you can probably use your favorite provider immediately instead of a self-hosted model. However, the point is, we have the options available and after all, a locally self-hosted model is the one we can trust the most.

### Configuration

Pi can mostly be configured using
- the `/settings` command,
- the `/model` command, and
- some of its keyboard shortcuts (which you can find with `/hotkeys`).
However, some settings are not exposed here and require us to manually write them.

`PI_CODING_AGENT_DIR`, by default `~/.pi/agent`, specifies where Pi saves its settings and other state files.

There are two main files we will consider, both located in `PI_CODING_AGENT_DIR`:
- `settings.json` is mainly configurable with the commands above, but there may be configuration values that are not exposed.
- `models.json` is used to list custom providers and the models that they make available.

#### Custom providers

To specify a custom provider in `models.json` (see above), we need, as a bare minimum,
1. the base URL of its API endpoint, usually something like `https://api.site.name/v1`,
2. the API key to authenticate us, or any value if we don't need an API key, and
3. the ID of the model that we want to access.

An example `model.json` that only contains a single locally running AI (i.e., running on your machine, not hosted somewhere else, in this example with [`llama.cpp`](https://llama.app/)) could look like this:

```json
{
  "providers": {
    "llama.cpp": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-responses",
      "apiKey": "dummy",
      "models": [{"id": "llama.cpp"}]
    }
  }
}
```

Explanation:
- `llama.cpp` is an arbitrary name for the provider. You can pick how you name your provider yourself.
- `baseUrl` is the aforementioned base URL of the API endpoint.
- `api` indicates how the provider communicates with us. Almost every provider supports `openai-completions`, but we prefer the newer `openai-responses` here.
- `apiKey` is the aforementioned API key for authenticating. You can specify it literally, as an environment variable using `$` (e.g., `$MY_API_KEY`), or by obtaining it from a shell command by prefixing the command `!`.
- `models` is a list of model configurations. The minimum requirement is, as you can see, a single model `id`.

However, we can also give Pi some additional information about the model, such as its maximum context length, whether it can use images as inputs, etc. For example, a model configuration for our Qwen3.6-27B model could look like the following. You would insert this as one entry in the `models` list:

```json
{
  "id": "Qwen/Qwen3.6-27B",
  "name": "Qwen3.6-27B",
  "reasoning": true,
  "input": ["text", "image"],
  "contextWindow": 262144,
  "maxTokens": 81920
}
```

Explanation:
- `id` is how the model is exposed by the provider; it needs to match, so that the provider knows which model we're trying to access (providers often host multiple models, not just one). We configured the provider so both the FP8 and BF16 version use the same ID, so it does not matter which version you are hosting.
- `name` is how the model is displayed to us in human-readable form.
- `reasoning` specifies whether the model can use reasoning to "think" before answering.
- `input` lets us specify the input modalities of the model. Since Qwen3.6-27B is multimodal, we allow text and images.
- `contextWindow` is the maximum context length that the model supports.
- `maxTokens` is the maximum number of total output tokens (i.e., both thinking and non-thinking tokens) the model can use for a single response. You can leave this setting out and let it default to 16384 if you don't know it, but for Qwen3.6-27B, we have a documented suggestion by the model authors, so we set it.
