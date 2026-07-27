import os
from minio import Minio
from minio.error import S3Error

class MinIOClient:
    def __init__(self, endpoint: str = None, access_key: str = None, secret_key: str = None, secure: bool = None, timeout: int = 30):
        """
        Αρχικοποιεί τον MinIO Client. 
        Αν δεν δοθούν ορίσματα, διαβάζει αυτόματα από τα Environment Variables (.env).
        """
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.timeout = timeout
        
        if secure is None:
            # Μετατροπή του string "False"/"True" από το .env σε boolean
            secure_str = os.getenv("MINIO_SECURE", "False").lower()
            self.secure = secure_str in ("true", "1")
        else:
            self.secure = secure

        print(f"[MinIO Config] endpoint={self.endpoint}, secure={self.secure}, timeout={self.timeout}s")

        # Δημιουργία του εσωτερικού Minio instance
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                http_client=None
            )
            self._test_connection()
        except Exception as e:
            print(f"[ERROR] Failed to initialize MinIO client: {e}")
            raise

    def _test_connection(self):
        """Test connectivity to MinIO server"""
        try:
            # Try to list buckets as a connectivity test
            buckets = self.client.list_buckets()
            print(f"[OK] Connected to MinIO. Found {len(buckets)} existing buckets.")
            return True
        except Exception as e:
            print(f"[WARNING] Connection test failed: {type(e).__name__}: {str(e)[:200]}")
            return False

    def upload_file(self, file_path: str, bucket_name: str, object_name: str = None) -> bool:
        """
        Ανεβάζει ένα αρχείο στο MinIO bucket. Αν το bucket δεν υπάρχει, το δημιουργεί.
        """
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            print(f"[UPLOAD] Uploading {object_name} ({file_size} bytes) to bucket '{bucket_name}'...")
            
            # Έλεγχος και αυτόματη δημιουργία bucket αν λείπει
            try:
                if not self.client.bucket_exists(bucket_name):
                    print(f"[BUCKET] Creating bucket '{bucket_name}'...")
                    self.client.make_bucket(bucket_name)
                    print(f"[OK] Created bucket: '{bucket_name}'")
            except Exception as e:
                print(f"[WARNING] Error checking/creating bucket: {e}")
                return False
            
            with open(file_path, 'rb') as file_data:
                self.client.put_object(
                    bucket_name,
                    object_name,
                    file_data,
                    length=file_size
                )
            
            print(f"[OK] Successfully uploaded {object_name} to '{bucket_name}'")
            return True
        
        except S3Error as e:
            print(f"[ERROR] MinIO S3 error: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Error uploading file: {type(e).__name__}: {str(e)[:500]}")
            return False
        
