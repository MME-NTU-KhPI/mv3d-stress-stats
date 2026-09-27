"""
Data access layer for MatViz3D / ANSYS HDF5 simulation results.

Wraps pymv3d and provides vectorized access to stress tensor fields,
spatial coordinate grids, load steps, and run metadata.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np


@dataclass
class RVEStressData:
    """Stress tensor field and spatial metadata for a single load step of an RVE."""
    file_path: str
    set_id: str
    load_step: int
    stress: np.ndarray             # shape (N, 6) or (Nx, Ny, Nz, 6)
    coordinates: Optional[np.ndarray] = None  # shape (N, 3) or (Nx, Ny, Nz, 3)
    grid_shape: Optional[Tuple[int, int, int]] = None
    applied_strain: Optional[np.ndarray] = None # macro strain (6,)
    seqv_ansys: Optional[np.ndarray] = None     # ANSYS/MatViz3D-computed SEQV if available (N,)
    metadata: Optional[Dict[str, Any]] = None


class RVEReader:
    """Reader for an individual MatViz3D HDF5 file."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = str(Path(file_path).resolve())
        if not Path(self.file_path).is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as f:
            # Set groups are typically "1", "2", ...
            # Filter out non-numeric metadata datasets like "last_set"
            self.set_ids: List[str] = [
                k for k in f.keys()
                if isinstance(f[k], h5py.Group) and k != "last_set"
            ]
            self.set_ids.sort(key=lambda s: int(s) if s.isdigit() else s)

    def __len__(self) -> int:
        return len(self.set_ids)

    def load_steps(self, set_index: int = 0) -> List[int]:
        """List available load step indices (1-based) for the specified set."""
        if not self.set_ids:
            return []
        set_id = self.set_ids[set_index]
        with h5py.File(self.file_path, "r") as f:
            grp = f[set_id]
            ls_keys = [k for k in grp.keys() if k.startswith("ls_")]
            # Extract integers: "ls_1" -> 1
            steps = []
            for k in ls_keys:
                try:
                    steps.append(int(k.split("_")[1]))
                except (IndexError, ValueError):
                    pass
            steps.sort()
            return steps

    def get_metadata(self, set_index: int = 0) -> Dict[str, Any]:
        """Extract solver, geometry, algorithm, and simulation parameters."""
        if not self.set_ids:
            return {}
        set_id = self.set_ids[set_index]
        meta: Dict[str, Any] = {
            "file_path": self.file_path,
            "set_id": set_id,
        }
        with h5py.File(self.file_path, "r") as f:
            grp = f[set_id]
            for field in ["cubeSize", "numPoints", "seed", "algorithm", "solver", "iterations_total"]:
                if field in grp:
                    val = grp[field][()]
                    if isinstance(val, (bytes, np.bytes_)):
                        val = val.decode("utf-8", errors="replace").rstrip("\x00")
                    elif isinstance(val, np.ndarray) and val.dtype in (np.int8, np.uint8):
                        val = bytes(val).decode("utf-8", errors="replace").rstrip("\x00")
                    meta[field] = val
        return meta

    def results(self, set_index: int = 0, load_step: int = 1) -> np.ndarray:
        """Raw per-voxel table from results dataset, shape (N, n_cols)."""
        if not self.set_ids:
            raise ValueError(f"No set groups found in {self.file_path}")
        set_id = self.set_ids[set_index]
        with h5py.File(self.file_path, "r") as f:
            ls_key = f"{set_id}/ls_{load_step}"
            if ls_key not in f:
                raise KeyError(f"Load step '{ls_key}' not found in {self.file_path}")
            return f[ls_key]["results"][:]

    def read_stress(
        self,
        set_index: int = 0,
        load_step: int = 1,
        as_grid: bool = False,
    ) -> RVEStressData:
        """Read 6-component Cauchy stress field for a given set and load step.
        
        Args:
            set_index: 0-based index of the set group (default 0 -> first group "1").
            load_step: 1-based load step number (e.g. 1, 2, ...).
            as_grid: If True, reshapes/scatters stress into (Nx, Ny, Nz, 6) 3D spatial grid.
                     If False, returns flat spatial samples of shape (N, 6).
                     
        Returns:
            RVEStressData containing stress array, coordinates, and metadata.
        """
        if not self.set_ids:
            raise ValueError(f"No set groups found in {self.file_path}")
        set_id = self.set_ids[set_index]

        with h5py.File(self.file_path, "r") as f:
            ls_key = f"{set_id}/ls_{load_step}"
            if ls_key not in f:
                raise KeyError(f"Load step '{ls_key}' not found in {self.file_path}")

            ls_grp = f[ls_key]
            results = ls_grp["results"][:]  # shape (N, 22)
            
            applied_strain = None
            if "eps_as_loading" in ls_grp:
                applied_strain = ls_grp["eps_as_loading"][:]

            # Coordinates: X (col 1), Y (col 2), Z (col 3)
            coords = results[:, 1:4]
            # Stresses: SX, SY, SZ, SXY, SYZ, SXZ (cols 7 to 12 inclusive)
            stress_flat = results[:, 7:13].astype(np.float64)
            seqv_ansys = results[:, 20].astype(np.float64) if results.shape[1] > 20 else None

            meta = self.get_metadata(set_index)
            cube_size = meta.get("cubeSize", None)

            if as_grid:
                # Map coordinates onto regular grid
                ix = coords[:, 0].astype(np.intp)
                iy = coords[:, 1].astype(np.intp)
                iz = coords[:, 2].astype(np.intp)
                
                nx = int(ix.max()) + 1 if len(ix) else 0
                ny = int(iy.max()) + 1 if len(iy) else 0
                nz = int(iz.max()) + 1 if len(iz) else 0
                
                grid_stress = np.zeros((nx, ny, nz, 6), dtype=np.float64)
                grid_stress[ix, iy, iz, :] = stress_flat
                
                grid_coords = np.zeros((nx, ny, nz, 3), dtype=np.float64)
                grid_coords[ix, iy, iz, :] = coords
                
                if seqv_ansys is not None:
                    grid_seqv = np.zeros((nx, ny, nz), dtype=np.float64)
                    grid_seqv[ix, iy, iz] = seqv_ansys
                else:
                    grid_seqv = None
                    
                return RVEStressData(
                    file_path=self.file_path,
                    set_id=set_id,
                    load_step=load_step,
                    stress=grid_stress,
                    coordinates=grid_coords,
                    grid_shape=(nx, ny, nz),
                    applied_strain=applied_strain,
                    seqv_ansys=grid_seqv,
                    metadata=meta,
                )
            else:
                return RVEStressData(
                    file_path=self.file_path,
                    set_id=set_id,
                    load_step=load_step,
                    stress=stress_flat,
                    coordinates=coords,
                    grid_shape=None,
                    applied_strain=applied_strain,
                    seqv_ansys=seqv_ansys,
                    metadata=meta,
                )
