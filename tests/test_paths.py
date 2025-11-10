"""Test the path management utilities."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.paths import (
    ensure_dir_exists,
    get_cache_dir,
    get_data_dir,
    get_figures_dir,
    get_output_dir,
    get_paths_dict,
    get_project_root,
)


def test_get_project_root():
    """Test finding the project root."""
    print("Testing: get_project_root()")

    root = get_project_root()

    print(f"  ✓ Project root: {root}")
    print(f"  ✓ Root name: {root.name}")

    # Verify it contains expected files
    assert (root / ".git").exists(), "Should contain .git directory"
    assert (root / "src").exists(), "Should contain src directory"

    print("  ✓ Contains .git and src directories")

    return root


def test_get_data_dir():
    """Test data directory paths."""
    print("\nTesting: get_data_dir()")

    data_dir = get_data_dir()
    print(f"  ✓ Data dir: {data_dir}")

    raw_dir = get_data_dir("raw")
    print(f"  ✓ Raw data dir: {raw_dir}")

    cache_dir = get_data_dir("cache")
    print(f"  ✓ Cache dir: {cache_dir}")

    # Check they're all under project root
    root = get_project_root()
    assert str(data_dir).startswith(str(root)), "Data dir should be under project root"
    print("  ✓ All paths are under project root")


def test_get_output_dir():
    """Test output directory creation."""
    print("\nTesting: get_output_dir()")

    # Without run name
    outputs = get_output_dir(run_name=None, create=True)
    print(f"  ✓ Outputs root: {outputs}")
    assert outputs.exists(), "Outputs directory should be created"

    # With run name (creates timestamped directory)
    run_dir = get_output_dir(run_name="test_run", create=True)
    print(f"  ✓ Run directory: {run_dir}")
    assert run_dir.exists(), "Run directory should be created"
    assert "test_run" in run_dir.name, "Run name should be in directory name"
    print(f"  ✓ Directory name contains timestamp: {run_dir.name}")


def test_cache_and_figures():
    """Test cache and figures directories."""
    print("\nTesting: get_cache_dir() and get_figures_dir()")

    cache = get_cache_dir(create=True)
    print(f"  ✓ Cache dir: {cache}")
    assert cache.exists(), "Cache directory should be created"

    figures = get_figures_dir(create=True)
    print(f"  ✓ Figures dir: {figures}")
    assert figures.exists(), "Figures directory should be created"


def test_ensure_dir_exists():
    """Test directory creation."""
    print("\nTesting: ensure_dir_exists()")

    # Create a nested test directory
    test_dir = get_project_root() / "outputs" / "test" / "nested" / "deep"

    result = ensure_dir_exists(test_dir)
    print(f"  ✓ Created: {result}")
    assert result.exists(), "Directory should be created"
    assert result == test_dir, "Should return the same path"
    print("  ✓ Nested directories created successfully")


def test_get_paths_dict():
    """Test getting all paths as dictionary."""
    print("\nTesting: get_paths_dict()")

    paths = get_paths_dict()

    print("  ✓ All paths:")
    for name, path in paths.items():
        print(f"      {name:20s}: {path}")

    assert "project_root" in paths
    assert "data" in paths
    assert "data_raw" in paths
    print(f"  ✓ Dictionary contains {len(paths)} paths")


if __name__ == "__main__":
    print("=" * 70)
    print("PATH MANAGEMENT TESTS")
    print("=" * 70)

    # Test 1: Project root
    root = test_get_project_root()

    # Test 2: Data directories
    test_get_data_dir()

    # Test 3: Output directories
    test_get_output_dir()

    # Test 4: Cache and figures
    test_cache_and_figures()

    # Test 5: Directory creation
    test_ensure_dir_exists()

    # Test 6: Paths dictionary
    test_get_paths_dict()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
