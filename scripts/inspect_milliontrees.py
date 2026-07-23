"""
Inspect MillionTrees package.

This script inspects the installed MillionTrees package
without assuming the internal API.
"""

from milliontrees import supported_datasets
from milliontrees import get_dataset
import inspect

print("=" * 60)
print("MillionTrees inspection")
print("=" * 60)

print("\nSupported datasets:")
print(supported_datasets)

print("\nFunction signature:")
print(inspect.signature(get_dataset))

print("\nTrying TreePolygons...")

try:

    dataset = get_dataset("TreePolygons")

    print("\nSUCCESS!")

    print("Dataset type:", type(dataset))

    try:
        print("Length:", len(dataset))
    except Exception as e:
        print("Cannot compute length:", e)

    try:
        sample = dataset[0]

        print("\nSample type:", type(sample))

        if hasattr(sample, "keys"):
            print("Sample keys:")
            print(sample.keys())
        else:
            print(sample)

    except Exception as e:
        print("Cannot access first sample:", e)

except Exception as e:

    print("\nERROR while loading dataset:")
    print(type(e).__name__)
    print(e)