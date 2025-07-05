"""
Utility functions for handling configuration in Docker environments.
Uses Docker mounted files for secure configuration management.
"""
import os


def get_config_value(key_name: str) -> str:
    """
    Get a configuration value from mounted config files.
    
    Args:
        key_name: Name of the config key (file will be at /run/secrets/{key_name})
        
    Returns:
        Configuration value as string
        
    Raises:
        ValueError: If config is not found or cannot be read
    """
    config_path = f"/run/secrets/{key_name}"
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration '{key_name}' not found at {config_path}")
    
    try:
        # Try different encodings and handle BOM
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                with open(config_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    # Remove any remaining BOM or control characters
                    content = content.lstrip('\ufeff\x00\xff\xfe')
                    # Remove all null bytes (common with UTF-16 misreading)
                    content = content.replace('\x00', '')
                    # Handle KEY=value format
                    if '=' in content:
                        return content.split('=', 1)[1].strip()
                    return content
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode file with any encoding")
    except Exception as e:
        raise ValueError(f"Could not read configuration '{key_name}': {e}")


def get_api_key() -> str:
    """
    Get the authentication key from environment variable or file.
    
    Returns:
        Authentication key as string
        
    Raises:
        ValueError: If key is not found
    """
    # Try environment variable first
    for env_var in ['GEMINI_API_KEY', 'API_KEY']:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key.strip()
    
    # Try Docker secrets path
    docker_secrets_path = "/run/secrets/gemini_api_key"
    if os.path.exists(docker_secrets_path):
        try:
            # Try different encodings and handle BOM
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    with open(docker_secrets_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                        # Remove any remaining BOM or control characters
                        content = content.lstrip('\ufeff\x00\xff\xfe')
                        # Remove all null bytes (common with UTF-16 misreading)
                        content = content.replace('\x00', '')
                        # Handle KEY=value format
                        if '=' in content:
                            return content.split('=', 1)[1].strip()
                        return content
                except UnicodeDecodeError:
                    continue
        except Exception:
            pass
    
    # Try local creds file
    local_creds_path = "/app/creds/api.txt"
    if os.path.exists(local_creds_path):
        try:
            with open(local_creds_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Handle KEY=value format
                if '=' in content:
                    return content.split('=', 1)[1].strip()
                return content
        except Exception:
            pass
    
    # Try relative creds file (for development)
    relative_creds_path = "creds/api.txt"
    if os.path.exists(relative_creds_path):
        try:
            with open(relative_creds_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Handle KEY=value format
                if '=' in content:
                    return content.split('=', 1)[1].strip()
                return content
        except Exception:
            pass
    
    raise ValueError("Auth key not found. Please set API_KEY environment variable or place key in creds/api.txt") 