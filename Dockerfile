FROM public.ecr.aws/lambda/python:3.10

# Install system packages
RUN yum install -y git

# Copy app code
COPY app.py wsgi_handler.py requirements.txt ./

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set handler
CMD ["wsgi_handler.handler"]
