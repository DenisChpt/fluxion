#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
	F32,
	F64,
}

impl DType {
	#[inline]
	#[must_use]
	pub const fn size_bytes(self) -> usize {
		match self {
			Self::F32 => 4,
			Self::F64 => 8,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn size_bytes_correct() {
		assert_eq!(DType::F32.size_bytes(), 4);
		assert_eq!(DType::F64.size_bytes(), 8);
	}

	#[test]
	fn equality_and_hash() {
		use std::collections::HashSet;
		let mut set = HashSet::new();
		set.insert(DType::F32);
		set.insert(DType::F64);
		set.insert(DType::F32); // duplicate
		assert_eq!(set.len(), 2);
	}

	#[test]
	fn copy_semantics() {
		let a = DType::F64;
		let b = a;
		assert_eq!(a, b);
	}
}
