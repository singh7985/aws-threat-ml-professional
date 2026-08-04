FROM public.ecr.aws/lambda/python:3.12

COPY services/scorer/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir \
    --requirement ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target ${LAMBDA_TASK_ROOT}

COPY models/model.joblib ${LAMBDA_TASK_ROOT}/models/model.joblib
COPY models/feature_manifest.json ${LAMBDA_TASK_ROOT}/models/feature_manifest.json
COPY src/threat_ml ${LAMBDA_TASK_ROOT}/threat_ml
COPY services/scorer/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.handler"]
