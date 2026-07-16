

## FlagTree Test-Report

FlagTree tests are validated on different backends, but currently the tests consist of only unit tests, which we will refine in the future for smaller or larger scale tests.

### 1. Python unit test:

| 　　　　　　　　　　　　 | default                   | iluvatar                                 | klx xpu                                       | mthreads                                       | metax                                       | hcu                                       |
|----------------------|---------------------------|-------------------------------------------|------------------------------------------------|------------------------------------------------|---------------------------------------------|---------------------------------------------|
| Number of unit tests | 9161 items               | 11395 items                               | 4183 items                                    | 4116 items                                    | 6309 items                                 | 309 items                                 |
| Script location      | flagtree/python/test/unit | flagtree/third_party/iluvatar/python/test/unit | flagtree/third_party/xpu/python/test/unit | flagtree/third_party/mthreads/python/test/unit | flagtree/third_party/metax/python/test/unit | flagtree/third_party/hcu/python/test/unit |
| Test command         | python3 -m pytest -s      | python3 -m pytest -s                      | python3 -m pytest -s                           | python3 -m pytest -s                           | python3 -m pytest -s                        | sh flagtree_test.sh                        |
| Passing rate         | 100%                      | 100%                                      | 100%                                           | 100%                                           | 100%                                        | 100%                                        |
