pub mod backend;
pub mod boundary;
pub mod dtype;
pub mod error;
pub mod grid;
pub mod operator;
pub mod stencil;

pub use backend::{Backend, BackendBuffer};
pub use boundary::{Boundaries, BoundaryCondition};
pub use dtype::DType;
pub use error::{CoreError, Result};
pub use grid::Grid;
pub use operator::{OperatorKind, OperatorSpec};
pub use stencil::{Stencil, StencilEntry};
