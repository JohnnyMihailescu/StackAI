"""
Cleanup script to remove libraries created by Locust load testing.

Usage:
    python scripts/cleanup_loadtest.py
    python scripts/cleanup_loadtest.py --host http://localhost:8000
    python scripts/cleanup_loadtest.py --dry-run  # Preview what would be deleted
"""

import argparse
import httpx


def cleanup(host: str, dry_run: bool = False):
    """Delete all libraries with 'loadtest_' prefix."""
    client = httpx.Client(base_url=host, timeout=30.0)

    # Get all libraries
    response = client.get("/api/v1/libraries")
    response.raise_for_status()
    libraries = response.json()

    # Filter to loadtest libraries
    loadtest_libs = [lib for lib in libraries if lib["id"].startswith("loadtest_")]

    if not loadtest_libs:
        print("No loadtest libraries found.")
        return

    print(f"Found {len(loadtest_libs)} loadtest libraries")

    if dry_run:
        print("\nDry run - would delete:")
        for lib in loadtest_libs:
            print(f"  - {lib['id']}: {lib['name']}")
        return

    # Delete each library
    deleted = 0
    failed = 0
    for lib in loadtest_libs:
        try:
            response = client.delete(f"/api/v1/libraries/{lib['id']}")
            if response.status_code == 204:
                deleted += 1
                print(f"  Deleted: {lib['id']}")
            else:
                failed += 1
                print(f"  Failed: {lib['id']} (status {response.status_code})")
        except Exception as e:
            failed += 1
            print(f"  Error: {lib['id']} - {e}")

    print(f"\nDone. Deleted: {deleted}, Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup loadtest data")
    parser.add_argument("--host", default="http://localhost:8000", help="API host")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    cleanup(args.host, args.dry_run)
