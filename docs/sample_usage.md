Sample images
The samples folder contains small images for end-to-end testing.

- samples/bird.png and samples/drone.png are example bird and drone photos.
- samples/sample_1.jpg through sample_6.jpg are mixed samples.

Use these images to exercise the API without a real deployment.

Uploading an image:
Send a POST request to /api/detect with a file field named image.

Example using curl:
curl -X POST http://127.0.0.1:8000/api/detect -F image=samples/drone.png

A successful response includes decision, confidence, reason, processing_time_ms, models, result, and uploaded_image.

