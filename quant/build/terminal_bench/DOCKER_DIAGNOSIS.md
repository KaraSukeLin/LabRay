# 目前環境的 Docker 診斷（2026-10-21）

本次於 2026-10-21 01:51–01:53 UTC 重新實測。原始證據在 [diagnostics/docker_20261021/](diagnostics/docker_20261021/)。只讀取環境、執行短暫子 namespace 探測；未安裝套件、啟動 daemon、修改安全政策或進行 GPU 推論。官方 Smoke 仍為 0／10，正式實驗維持 PAUSED。

**目前工作環境沒有提供 Docker 執行能力。阻礙包含未安裝軟體、沒有 daemon 連線，以及外層容器的權限與資源管理限制。單純安裝 Docker CLI 無法補齊這些條件。** 這不代表 Terminal-Bench 必須使用本機 Docker；Harbor 也支援其他任務 sandbox。

## 已確認的事實

| 層次 | 實際證據 | 影響 |
|---|---|---|
| Docker 程式 | `docker version` 回傳 127：`docker: command not found`；PATH、標準執行檔位置及 dpkg 套件紀錄均未找到 Docker／dockerd／Compose | CLI 及 Engine 沒有在目前環境中提供；不是單純 daemon 停止 |
| Daemon 連線 | `/var/run/docker.sock`、`/run/docker.sock`、各已檢查 rootless socket 均不存在；`DOCKER_HOST`／`DOCKER_CONTEXT` 未設定，沒有本機 Docker context 設定 | 沒有已提供的本機或遠端 daemon 入口；也未見 dockerd 程序。無法由此判定宿主機是否另有 Docker |
| 一般 Docker 所需權限 | UID=0，但有效與 bounding capabilities 都沒有 `CAP_SYS_ADMIN`／`CAP_NET_ADMIN`；直接建立 mount／net namespace 回傳 `Operation not permitted` | 現有 root 並非具備完整宿主機權限的 root；不能靠一般 sudo／重新執行程式取得這些缺少的外層權限 |
| 掛載操作 | 新建 user＋mount namespace 後，tmpfs mount 仍回傳 `EACCES`；該子 namespace 的 `CapEff=000001ffffffffff`，`Seccomp=0` | 能建立 namespace 不等於可以執行容器所需掛載；rootless 路徑仍有實質阻礙 |
| CPU／RAM 控制 | `/sys/fs/cgroup` 是唯讀 cgroup v2，`cgroup.subtree_control` 為空，沒有可用的 user systemd manager | 目前沒有可用的子 cgroup 委派路徑，不能證明每個官方任務的 CPU／RAM 限制可被落實 |
| Rootless 工具與身份映射 | 缺少 `newuidmap`／`newgidmap`、RootlessKit 等工具；`/etc/subuid`／`subgid` 只有 `coder:100000:65536`，目前 root 沒有配置 | 不符合目前使用者的 Docker rootless 安裝先決條件；改用 coder 也不會自動解決掛載及 cgroup 限制 |

可見環境是 Debian 12、kernel `5.15.0-94-generic`；PID 1 為 `supervisord`，`systemd-detect-virt --container` 回傳 `container-other`。根目錄使用 containerd snapshotter 的 overlay，程序 AppArmor profile 是 `cri-containerd.apparmor.d (enforce)`。這些證據符合受管理的 containerd 容器環境，**不代表已確認租用平台、宿主機部署方式或宿主機 Docker 狀態**。

目前可見 CPU 上限為 50 cores，RAM 上限為 1,000 GiB。這是整個外層環境的上限，不是各 Terminal-Bench 任務的獨立資源限制。本次五張 H100 均可由 `nvidia-smi` 讀取、每卡 1 MiB／0% 使用；GPU 可用不會補足容器執行權限。

## Namespace 與 AppArmor 的關鍵差異

| 實測命令 | Exit code | 結果 |
|---|---:|---|
| `unshare --user /bin/true` | 0 | user namespace 可建立 |
| `unshare --mount /bin/true` | 1 | 缺外層權限，`Operation not permitted` |
| `unshare --net /bin/true` | 1 | 缺外層權限，`Operation not permitted` |
| `unshare --user --map-root-user --net /bin/true` | 0 | 新 user namespace 下的 net namespace 可建立 |
| `unshare --user --map-root-user --mount --net --propagation unchanged /bin/true` | 0 | 不改 propagation 時，三種 namespace 可一起建立 |
| `unshare --user --map-root-user --mount /bin/true` | 1 | namespace 建立後，變更根掛載 propagation 遭拒：`Permission denied` |
| 在上述獨立 user＋mount namespace 內呼叫 `mount(tmpfs)` | 1 | `errno=13 (EACCES)`；臨時目錄已移除，沒有留下掛載 |

子 namespace 保留 `cri-containerd.apparmor.d (enforce)`，即使取得該 user namespace 的完整 capabilities，掛載仍失敗。containerd 官方預設 AppArmor template 含有 `deny mount,`，與觀察到的行為高度一致。因此，**外層 AppArmor 是掛載遭拒的高可信解釋**。[containerd 官方 template](https://github.com/containerd/containerd/blob/v1.7.19/contrib/apparmor/template.go)

但宿主機實際載入的 profile 內容、containerd 版本及 kernel audit log 在此容器內不可見；這次沒有卸載／修改 profile 做對照，故不能聲稱已從宿主機 audit 確認每一次拒絕都由該規則造成。引用 v1.7.19 是說明官方預設規則，不是宣稱宿主機使用該版本。

診斷工具的預設命令 sandbox 也回傳 `bwrap: Failed to make / slave: Permission denied`；本次唯讀檢查改由獲准的執行方式完成。這是工具啟動時的掛載失敗，**不是 Docker daemon 本身的 log**，不與 `docker: command not found` 混為一談。被檢查程序顯示 `Seccomp=0`，因此沒有證據將本次失敗歸因於該程序的 seccomp filter。

## 為何 rootless 也不能直接解決

Docker rootless 的官方先決條件包含 `newuidmap`／`newgidmap` 和目前使用者至少 65,536 個 subordinate UIDs／GIDs；rootless 仍要建立和管理容器掛載，並非只需要 `unshare --user` 成功。[Docker rootless 說明](https://docs.docker.com/engine/security/rootless/)

此外，Docker 官方文件指出，rootless 模式的 cgroup 資源限制需具備 cgroup v2、systemd 及相應 controller 委派；條件不足時，`--cpus`／`--memory` 等參數可能被忽略。當前既無 user systemd manager，cgroup 掛載又唯讀，不能用「容器成功啟動」代替「官方任務資源限制正確生效」。[Docker rootless 資源限制](https://docs.docker.com/engine/security/rootless/tips/#limiting-resources)

本次未安裝或啟動 rootless Docker，不宣稱測得 dockerd 啟動錯誤；也不宣稱所有其他 runtime 永遠不可行。已實測的是：目前必要工具不齊、掛載遭拒、沒有可用的資源委派。只換 Podman 或補裝 CLI 不會自動消除這些外層限制。

## 處理方式與責任位置

| 路徑 | 需要補足的條件 | 目前狀態 |
|---|---|---|
| 由平台提供既有 Docker daemon | 平台提供可存取的 socket／受保護遠端入口；環境內有 CLI＋Compose；daemon 看得到 Harbor 使用的 bind-mount 工作路徑，並能落實任務資源限制 | 最能沿用目前已準備的 Docker 配置；需要平台管理者協助 |
| 在此容器內啟動巢狀 Docker | 由平台提供正式支援巢狀容器的權限、安全 profile、掛載及 cgroup 配置，再安裝 Engine／Compose | 目前不具備；不是在容器內執行 `apt install` 就能完成 |
| 使用 Harbor 雲端 sandbox，例如 Daytona | 帳號與憑證、網路、費用；逐項核對固定官方 image、資源、timeout、Verifier 與資料擷取；現有 runner 需做最小後端適配 | 官方支援此方向，但本專案尚未實測；不必將五張 H100 的模型推論移走 |

Docker 官方支援透過 Unix socket 或設定遠端 daemon 連線；遠端 daemon 的 bind mount 路徑在 daemon 所在環境解析，因此不能只設定 `DOCKER_HOST` 就假定現有本機路徑可用。[Docker daemon 連線](https://docs.docker.com/engine/daemon/remote-access/)、[bind mount 說明](https://docs.docker.com/engine/storage/bind-mounts/)

Terminal-Bench 2.1 官方 README 本來就以 Harbor＋Daytona 示範雲端 sandbox。因此應將目前阻礙表述為「**選定的 Docker 後端不可用，尚無已驗證的替代任務 sandbox**」，不能表述為「Terminal-Bench 一定要本機 Docker」。[固定 revision 的官方說明](https://github.com/harbor-framework/terminal-bench-2-1/blob/7131e4375048a0e408a8fb404b5f499d726b695b/README.md)

可提供給平台管理者的問題描述：

> 目前只能使用容器內帳號，UID=0，但沒有 Docker CLI／dockerd／Compose，沒有掛載 docker.sock，也沒有遠端 daemon 設定。CapEff／CapBnd 缺 SYS_ADMIN 與 NET_ADMIN，cgroup v2 唯讀。AppArmor 為 cri-containerd.apparmor.d enforce；在新 user＋mount namespace 內，即使具備該 namespace 的完整 capabilities，tmpfs mount 仍回 EACCES。需要可執行 Harbor 官方任務映像、Docker Compose、工作目錄 bind mounts，以及各任務 CPU／RAM 限制的受支援執行方式。請確認能否提供 daemon 存取或支援巢狀容器的環境，並從宿主機 audit log 確認 mount 拒絕原因。

## 可重現檢查與證據

最小失敗重現（已重複執行）：

```sh
docker version
# exit 127: docker: command not found
```

沿用既有 harness 的檢查，不啟動 GPU 推論或任務容器：

```sh
python3 src/terminal_bench.py preflight \
  --output build/terminal_bench/diagnostics/docker_recheck
# 目前 exit 1，preflight.json 的 ready=false
```

| 原始檔 | 內容 |
|---|---|
| [evidence.json](diagnostics/docker_20261021/evidence.json) | 程式、套件、socket、capabilities、AppArmor、cgroup、UID/GID、systemd 與 GPU 唯讀盤點；包含各 probe 的 argv、return code、stdout、stderr |
| [namespace_probes.json](diagnostics/docker_20261021/namespace_probes.json) | 保持／變更 propagation 的對照，namespace 內 capabilities 與 AppArmor |
| [mount_probe.json](diagnostics/docker_20261021/mount_probe.json) | tmpfs 掛載的完整 probe 程式、EACCES 與清理紀錄 |
| [reproduction.json](diagnostics/docker_20261021/reproduction.json) | 再次執行 Docker 命令及原有 harness 的失敗結果；標準執行檔位置檢查 |
| [harness/preflight.json](diagnostics/docker_20261021/harness/preflight.json) | 原有 preflight 產生的 ready=false 與原始錯誤 |

本次診斷已完成；環境問題尚未修復。沒有新的 Docker／Harbor／推論背景服務，沒有執行任何 Benchmark trials，也沒有修改官方資料、模型或推論環境。未修復宿主機配置，故沒有宣稱故障重現已轉為成功。
