# Release Management

When creating a new release, make sure to update the following files:

- `CMakeLists.txt`

    ```markdown
    # The library version is versioned off the major version. If the API
    # changes, the library version should be bumped.
    set(PROJECT_VERSION_MAJOR "2")
    set(PROJECT_VERSION_MINOR "3")
    set(PROJECT_VERSION_PATCH "0")
    ```

- `docs/source/conf.py`

    ```python
    # The full version, including alpha/beta/rc tags
    release = 'v2.3.0'
    ```

Create an annotated tag:

```console
 git tag --annotate --sign --message='Release v2.3.0' v2.3.0
```

Or use the GitHub UI at <https://github.com/lanl/bml/releases> to create a new
release.

The changelog can be generated via

```console
 git log --oneline --no-decorate --no-merges v2.2.0..v2.3.0
 ```

 Or using the GitHub UI.
