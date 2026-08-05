FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV MODEL_DIRECTORY=/var/task/model

COPY services/scorer/requirements.txt \
    ${LAMBDA_TASK_ROOT}/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --requirement ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target ${LAMBDA_TASK_ROOT}

COPY src/threat_ml \
    ${LAMBDA_TASK_ROOT}/threat_ml

COPY services/scorer/handler.py \
    ${LAMBDA_TASK_ROOT}/handler.py

COPY models/model.joblib \
    ${LAMBDA_TASK_ROOT}/model/model.joblib

COPY models/feature_manifest.json \
    ${LAMBDA_TASK_ROOT}/model/feature_manifest.json

CMD ["handler.lambda_handler"]
