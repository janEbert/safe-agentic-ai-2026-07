# Installation instructions for safe agentic AI workshop

This is the software we'd ask you to install in advance for the safe agentic AI workshop on 2026-07-13.

Note that there are some differences depending on whether you are using Windows, macOS, or Linux. Please look out for these differences.

As a short summary, we'll be using NodeJS to install a sandbox runtime and an agentic harness.

## NodeJS

We'll be using packages developed in JavaScript for the course. [NodeJS](https://nodejs.org/en) provides packaging infrastructure allows us to run these without a browser. We prefer the latest version with all the newest features.

You do not need to _install_ NodeJS if you prefer not to; you can just download pre-built binaries that you can execute without having to install anything. Be aware that you may want to modify your system `PATH` to point at the location of the downloaded binaries, so that you can easily execute them from anywhere.

- The current latest version binaries can be found here, under "Binary Downloads" if you'd prefer not to install NodeJS:  
  https://nodejs.org/en/download/archive/v26.4.0
- Alternatively, the official and up-to-date website lists installation instructions and the pre-built binaries under "Or get a prebuilt Node.js® for [...]":  
  https://nodejs.org/en/download/current

## Package dependencies

### Windows

Windows users do not need additional dependencies.

### macOS

macOS users need to install [ripgrep](https://github.com/BurntSushi/ripgrep).

#### ripgrep

- Similar to NodeJS, you can download pre-built binaries:  
  https://github.com/BurntSushi/ripgrep/releases
- Or install the package according to the official instructions:  
  https://github.com/BurntSushi/ripgrep#installation

### Linux

Linux users need to install [ripgrep](https://github.com/BurntSushi/ripgrep), [bubblewrap](https://github.com/containers/bubblewrap), and [socat](http://www.dest-unreach.org/socat/).

#### ripgrep

- Similar to NodeJS, you can download pre-built binaries:  
  https://github.com/BurntSushi/ripgrep/releases
- Or install the package according to the official instructions:  
  https://github.com/BurntSushi/ripgrep#installation

#### Bubblewrap and socat

Please install these via your system's package manager. E.g., for Ubuntu:

```shell
sudo apt-get install bubblewrap socat
```

## NodeJS packages

Before we install the NodeJS packages, let's configure `npm` to only install packages that have been live for 3 days, in order to try to avoid [supply chain attacks](https://en.wikipedia.org/wiki/Supply_chain_attack).

```shell
npm config set min-release-age=3 ignore-scripts=true
```

### Sandbox Runtime

We install the [Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime) via NodeJS's `npm`.

```shell
npm install -g @anthropic-ai/sandbox-runtime
```

#### Windows

You need to execute the following after the above installation via `npm`:

```shell
npm exec @anthropic-ai/sandbox-runtime windows-install
```

Afterwards, log out and back in. If you see an error about a file not being accessible, try to navigate to the `%LocalAppData%\sandbox-runtime` directory to get a permission change pop-up. After confirming the permission change, rerun the above `npm exec` command and log out and back in.

#### macOS

Nothing else to do.

#### Linux

This applies only to certain Linux versions, for example, recent versions of Ubuntu, Debian, or RHEL.

We need to allow bubblewrap to use unprivileged user namespaces. We can achieve this by saving the following snippet to the file `/etc/apparmor.d/bwrap`:

```
# This profile allows everything and only exists to give the
# application a name instead of having the label "unconfined"

abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,

  # Site-specific additions and overrides. See local/README for details.
  include if exists <local/bwrap>
}
```

### Pi Agent

We install the [Pi](https://pi.dev/) agent harness via NodeJS's `npm`.

```shell
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```
