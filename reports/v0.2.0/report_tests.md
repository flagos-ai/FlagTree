

## FlagTree Test-Report

FlagTree tests are validated on different backends, but currently the tests consist of only unit tests, which we will refine in the future for smaller or larger scale tests.

### 1. Python unit test:

| 　　　　　　　　　　　　 | default                   | xpu (klx)                                 | iluvatar                                       | mthreads                                       | metax                                       |
|----------------------|---------------------------|-------------------------------------------|------------------------------------------------|------------------------------------------------|---------------------------------------------|
| Number of unit tests | 18518 items               | 12623 items                               | 14808 items                                    | 10392 items                                    | 13728 items                                 |
| Script location      | flagtree/python/test/unit | flagtree/third_party/xpu/python/test/unit | flagtree/third_party/iluvatar/python/test/unit | flagtree/third_party/mthreads/python/test/unit | flagtree/third_party/metax/python/test/unit |
| Test command         | python3 -m pytest -s      | python3 -m pytest -s                      | python3 -m pytest -s                           | python3 -m pytest -s                           | python3 -m pytest -s                        |
| Passing rate         | 100%                      | 100%                                      | 100%                                           | 100%                                           | 100%                                        |
