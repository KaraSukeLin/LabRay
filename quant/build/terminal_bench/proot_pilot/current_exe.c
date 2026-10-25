/* Narrow runtime shim: own executable only, from PRoot's AT_EXECFN.
 * No host proc filesystem is exposed. All other readlink calls are forwarded.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <unistd.h>

static int own_exe(const char *path) {
    return strcmp(path, "/proc/self/exe") == 0;
}

static ssize_t exe_link(char *buffer, size_t size) {
    int previous_errno = errno;
    if (!size) { errno = EINVAL; return -1; }
    if (!buffer) { errno = EFAULT; return -1; }
    const char *name = (const char *)getauxval(AT_EXECFN);
    char resolved[PATH_MAX];
    if (!name) { errno = ENOENT; return -1; }
    int found = 0;
    /* PRoot reports just "uvx" for a PATH launch; resolve it in that same
       guest PATH. This is scoped to the trusted verifier, not a procfs model. */
    if (!strchr(name, '/')) {
        char *paths = strdup(getenv("PATH") ? getenv("PATH") : "/usr/bin:/bin");
        char *cursor = paths, *part;
        while (paths && (part = strsep(&cursor, ":")) != NULL) {
            char candidate[PATH_MAX];
            int count = snprintf(candidate, sizeof(candidate), "%s/%s", *part ? part : ".", name);
            if (count > 0 && count < PATH_MAX && access(candidate, X_OK) == 0 && realpath(candidate, resolved)) { found = 1; break; }
        }
        free(paths);
    }
    if (!found && !realpath(name, resolved)) { errno = ENOENT; return -1; }
    size_t count = strlen(resolved);
    if (count > size) count = size;
    memcpy(buffer, resolved, count);
    errno = previous_errno;
    return (ssize_t)count;
}

ssize_t readlink(const char *path, char *buffer, size_t size) {
    if (own_exe(path)) return exe_link(buffer, size);
    ssize_t (*next)(const char *, char *, size_t) = dlsym(RTLD_NEXT, "readlink");
    return next(path, buffer, size);
}

ssize_t readlinkat(int dirfd, const char *path, char *buffer, size_t size) {
    if (own_exe(path)) return exe_link(buffer, size);
    ssize_t (*next)(int, const char *, char *, size_t) = dlsym(RTLD_NEXT, "readlinkat");
    return next(dirfd, path, buffer, size);
}
