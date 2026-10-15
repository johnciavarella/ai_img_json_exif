# Apple containers

Apple's own [`container`](https://github.com/apple/container) runs the app without Docker Desktop, one lightweight VM per container. `Containerfile` is the same one Docker builds from.

```sh
brew install container
container system start
./scripts/apple-container.sh up
```

Open <http://127.0.0.1:8090>. The script takes `build`, `up`, `down`, `ps`, `logs [web|vision|video]` and `url`, and reads the same `.env` as `compose.yaml`. Nothing else to set up: the script works out for itself how to reach a model server on the Mac (see **Host access** below), and that is the only part of `docker compose` this cannot take verbatim.

`container` behaves differently from `docker compose` in six ways, and each one needs handling:

| | Docker and Podman | Apple `container` |
| --- | --- | --- |
| Service addresses | `http://vision:8000`, `http://video:8001` | the containers' IP addresses, read at start |
| `/videos` | the `video-data` volume | a host folder, `video-data/` by default, bind mounted |
| Caches and saved workflows | volumes, writable straight away | volumes, given to the app's uid on first use |
| CPU and memory | one VM for everything | set per container by the script |
| Restart after a reboot | `restart: unless-stopped` | none; run `./scripts/apple-container.sh up` again |
| Compose | built in | none; the script takes its place |

- **Service addresses.** `container` resolves one container's name from another only when a DNS domain is registered ([apple/container#1809](https://github.com/apple/container/issues/1809)), and there is no `--add-host`. The script reads each service's IP after starting it and passes that to the web app. Container IPs change when a VM restarts, so run `up` again after a reboot.
- **One folder, not one volume.** A named volume can be attached to only one container at a time, and the web app and the video service both need `/videos`; the second one fails with `The storage device attachment is invalid`. A bind mount can be shared, so `/videos` is a host folder. Set `VIDEO_DATA=/Volumes/Media/video-data` in `.env` to move it off the boot disk, as you would for Docker.
- **Permissions.** Docker gives a fresh volume the ownership of the image folder it covers; `container` starts every volume empty and root-owned, and model loading then fails with `PermissionError: /cache/hf`. The script chowns `/cache` and `/data` to uid 1001 once, the first time.
- **Sizes.** `container` gives every container 4 CPUs and 1 GB of RAM unless told otherwise. SigLIP 2 alone needs about 1.5 GB, so the script gives the vision and video services every core and 8 GB each, and the web app 2. Lower them with `VISION_CPUS`, `VISION_MEMORY`, `VIDEO_CPUS`, `VIDEO_MEMORY`, `APP_CPUS` and `APP_MEMORY` in `.env`.
- **Builds.** `container build` runs in a helper container that defaults to 2 CPUs and 2 GB, which is thin for torch. `scripts/apple-container.sh build` widens it first.
- **Host access.** `host.containers.internal` is Podman's; Apple container has no such name, and the containers' own network is not the Mac's. A container reaches a model server on the Mac at the Mac's own address, so start it on every interface — `--host 0.0.0.0` in the [Quickstart](../README.md#prerequisite-local-vision-model) already does — and the script substitutes that address into `LLM_BASE_URL` for you, because `LLM_BASE_URL` defaults to `http://host.container.internal:8080/v1`. A `.env` that sets `LLM_BASE_URL` to anything else is left alone, so the `.env` you use with Docker carries over unchanged.
  - `host.container.internal` also works, as a DNS domain you register with `sudo container system dns create host.container.internal --localhost 203.0.113.113`. Two reasons not to: it needs a sudo password, and **it turns iCloud Private Relay off on your Mac**. It also has to be registered again after every reboot. To delete one you have registered: `sudo container system dns delete host.container.internal`.
  - Worth knowing which is which: the domain reaches a server bound to `127.0.0.1`, the Mac's address does not, and the Mac's address reaches a server bound to `0.0.0.0` from anywhere on your network rather than only from this Mac. Bind your model server to `0.0.0.0` only on a network you trust.
- **Speed.** Level with Docker Desktop, and 0.95× faster over all four vision models. See [Speed](#speed) below for the numbers and how they were taken.

[container-compose](https://github.com/mcrich23/container-compose) does read `compose.yaml`, and mostly works, but it drops `cpus`, `ulimits` and restart policies and patches hosts files to stand in for service names, so the script is the better path. `compose.yaml` now carries an explicit `name:`, which `container-compose` needs: it takes the project name from the folder otherwise, and prefixes volumes with an underscore, which `container` rejects when the folder itself starts with a `.` or `_`.

## Speed

Nothing about the models changes between runtimes — the same `Containerfile`, the same wheels, the same CPU. Measured on a 16-core Apple silicon Mac (macOS 27.0.1, 64 GB), over a folder of 12 photos, restarting the stack between every measurement and alternating the two runtimes so that whatever else the machine was doing hit both equally. Median of four rounds, per photo, in milliseconds:

| | Apple `container` | Docker Desktop | |
| --- | ---: | ---: | ---: |
| Read EXIF | 133 | 132 | 1.01× |
| EfficientNetV2-S | 265 | 254 | 1.05× |
| ResNet-50 | 152 | 144 | 1.06× |
| SigLIP 2 | 707 | 779 | **0.91×** |
| RT-DETRv2-S | 399 | 468 | **0.85×** |
| **All four models** | **1528** | **1603** | **0.95×** |
| Write metadata | 273 | 270 | 1.01× |

Apple `container` comes out ahead on the two heaviest models and level on the rest, 0.95× over all four. Results were identical, not just timings: the same category for every photo from every model, 716 EXIF tags read and 72 tags written per round, and no errors in any of the eight runs. Docker Desktop's VM had 8 CPUs and 7.6 GB, which is its default for this machine; the script gives each container every core and 8 GB.

Two caveats worth the space:

- **Give the machine a rest first.** Three VMs share the same physical cores, so if anything else on the Mac is busy both runtimes lose most of their speed, and the gap between them grows to two or three times over. The table above is what you get when the machine is otherwise idle, and the spread within each runtime is wide: ResNet-50 ranged from 79 to 161 ms on Apple `container` across four identical rounds.
- **Benchmark against a runtime you know.** Single runs are not comparable to each other on a busy machine. If you re-measure, alternate the two and compare medians, or the number you get will be an artefact of when it ran.

`./scripts/apple-container.sh up` publishes `127.0.0.1:8000` for the [CLI](cli.md), the same as `docker compose up`.
