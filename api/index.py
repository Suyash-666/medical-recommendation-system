from vercel_wsgi import handler as vercel_handler
from app import app

# Vercel expects a top-level "handler" callable for Python serverless functions.
# This wraps the existing Flask WSGI app.

def handler(event, context):
    return vercel_handler(app, event, context)
