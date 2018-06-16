#!/bin/bash

payload=$1
content=${2:-text/csv}

#curl --data-binary @${payload} -H "Content-Type: text/plain" -v http://localhost:8080/invocations
curl -d $payload -H "Content-Type: text/plain" -v http://localhost:8080/invocations

#curl --data "url=1905"  http://localhost:8080/invocations
