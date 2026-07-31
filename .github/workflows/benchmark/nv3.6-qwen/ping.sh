#!/bin/bash

export no_proxy="127.0.0.1,localhost,::1"
curl http://127.0.0.1:8000/v1/models
