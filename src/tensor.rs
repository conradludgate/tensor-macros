#[macro_export]
macro_rules! mul {
    () => (1);
    ($head:expr) => ($head);
    ($head:expr, $($tail:expr),+) => ($head * mul!($($tail),*));
}

#[macro_export]
macro_rules! sum {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),+) => (1 + sum!($($tail),*));
}

/// Generates a tensor type
///
/// Generates a type with the given name and dimensions (space seperated)
/// There's no upper limit on the amount of dimensions given
/// Matricies and vectors have special properties assigned to them
///
/// # Example
///
/// ```rust
/// #![feature(try_from)]
///
/// #[macro_use]
/// use tensor_macros::*;
/// use tensor_macros::traits::*;
///
/// tensor!(M23: 2 x 3);
///
/// assert_eq!(M23::<f64>::dims(), vec!(2, 3));
///
/// let m23: M23<f64> = Default::default();
/// assert_eq!(m23.get_dims(), vec!(2, 3));
/// ```
#[macro_export]
macro_rules! tensor {
	($name:ident: $dim:literal) => {
		make_tensor!($name $dim);

		impl<T: tensor_macros::traits::TensorTrait> tensor_macros::traits::Vector for $name<T> {
			const COLS: usize = $dim;
		}
	};

	($name:ident: row $dim:literal) => {
		make_tensor!($name $dim);

		impl<T: tensor_macros::traits::TensorTrait> tensor_macros::traits::RowVector for $name<T> {
			const ROWS: usize = $dim;
		}
	};

	($name:ident: $dim1:literal x $dim2:literal) => {
		make_tensor!($name $dim1 x $dim2);

		impl<T: tensor_macros::traits::TensorTrait> tensor_macros::traits::Matrix for $name<T> {
			const ROWS: usize = $dim1;
			const COLS: usize = $dim2;
		}
	};

	($name:ident: $($dim:literal)x+ ) => (
		make_tensor!($name $($dim) x *);
	)
}

/// Generates a tensor type
///
/// Use [`tensor!`] instead, it uses this macro and more
///
/// [`tensor!`]: macro.tensor.html
#[macro_export]
macro_rules! make_tensor {
	($name:ident $($dim:literal)x+ ) => {


		// #[derive(TensorTranspose)]
		pub struct $name<T: tensor_macros::traits::TensorTrait> ([T; mul!($($dim),*)]);

		// pub struct concat_idents!($name, _transpose)<T, TT>
		// 	where T: tensor_macros::traits::Tensor<TT>,
		// 	TT: tensor_macros::traits::TensorTrait
		// (T);

		impl<T: tensor_macros::traits::TensorTrait> $name<T> {
			#[allow(dead_code)]
			fn new() -> Self {
				Default::default()
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> tensor_macros::traits::Tensor for $name<T> {
			type Value = T;

			const SIZE: usize = mul!($($dim),*);
			const NDIM: usize = sum!($($dim),*);

			fn dims() -> Vec<usize> {
				vec!($($dim),*)
			}

			fn get_dims(&self) -> Vec<usize> {
				Self::dims()
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> Copy for $name<T> { }

		impl<T: tensor_macros::traits::TensorTrait> Clone for $name<T> {
			fn clone(&self)	-> Self {
				let mut data: [T; mul!($($dim),*)];

				unsafe {
					data = std::mem::uninitialized();

					for (i, elem) in (&mut data[..]).iter_mut().enumerate() {
						std::ptr::write(elem, self.0[i]);
				    }
				}

				$name::<T>(data)
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> PartialEq for $name<T> {
			fn eq(&self, other: &Self) -> bool {
				for (p, q) in self.0.iter().zip(other.0.iter()) {
					if p != q {
						return false;
					}
				}

				true
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::fmt::Debug for $name<T>  {
			fn fmt(&self, f:  &mut std::fmt::Formatter) -> std::fmt::Result {
				debug_tensor!(f, self; $($dim),*;);

				Ok(())
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> Default for $name<T> {
			fn default() -> Self {
				$name::<T>([Default::default(); mul!($($dim),*)])
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::convert::TryFrom<&[T]> for $name<T> {
			type Error = tensor_macros::traits::TensorError;

			fn try_from(v: &[T]) -> Result<Self, Self::Error> {
				if v.len() < mul!($($dim),*) {
					Err(tensor_macros::traits::TensorError::Size)
				} else {
					let mut a: [T; mul!($($dim),*)] = [Default::default(); mul!($($dim),*)];
					a.copy_from_slice(&v[..mul!($($dim),*)]);
					Ok($name::<T>(a))
				}
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::convert::TryFrom<Vec<T>> for $name<T> {
			type Error = tensor_macros::traits::TensorError;

			fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
				if v.len() < mul!($($dim),*) {
					Err(tensor_macros::traits::TensorError::Size)
				} else {
					let mut a: [T; mul!($($dim),*)] = [Default::default(); mul!($($dim),*)];
					a.copy_from_slice(&v[..mul!($($dim),*)]);
					Ok($name::<T>(a))
				}
			}
		}

		impl<T: tensor_macros::traits::TensorTrait> std::convert::From<T> for $name<T> {
			fn from(t: T) -> Self {
				$name::<T>([t; mul!($($dim),*)])
			}
		}

		impl<T, U, V> std::ops::Add<$name<U>> for $name<T>
			where Self: tensor_macros::traits::Tensor<Value=T>,
			T: tensor_macros::traits::TensorTrait + std::ops::Add<U, Output=V>,
			U: tensor_macros::traits::TensorTrait,
			V: tensor_macros::traits::TensorTrait,
		{
			type Output = $name<V>;

			fn add(self, other: $name<U>) -> Self::Output {
				let mut data: [V; mul!($($dim),*)];

				unsafe {
					data = std::mem::uninitialized();

					for (i, elem) in (&mut data[..]).iter_mut().enumerate() {
						std::ptr::write(elem, self.0[i] + other.0[i]);
				    }
				}

				$name::<V>(data)
		    }
		}

		impl<T, U> std::ops::AddAssign<$name<U>> for $name<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::AddAssign<U>,
			U: tensor_macros::traits::TensorTrait,
		{
			fn add_assign(&mut self, other: $name<U>) {
				for i in 0..mul!($($dim),*) {
					self[i] += other[i];
				}
		    }
		}

		impl<T, U, V> tensor_macros::traits::CwiseMul<$name<U>> for $name<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::Mul<U, Output=V>,
			U: tensor_macros::traits::TensorTrait,
			V: tensor_macros::traits::TensorTrait,
		{
			type Output = $name<V>;

			fn cwise_mul(self, other: $name<U>) -> Self::Output {
				let mut data: [V; mul!($($dim),*)];

				unsafe {
					data = std::mem::uninitialized();

					for (i, elem) in (&mut data[..]).iter_mut().enumerate() {
						std::ptr::write(elem, self.0[i] * other.0[i]);
				    }
				}

				$name::<V>(data)
			}
		}

		impl<T, U> tensor_macros::traits::CwiseMulAssign<$name<U>> for $name<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::MulAssign<U>,
			U: tensor_macros::traits::TensorTrait,
		{
			fn cwise_mul_assign(&mut self, other: $name<U>) {
				for i in 0..mul!($($dim),*) {
					self[i] *= other[i];
				}
			}
		}

		impl<T, U> std::ops::MulAssign<U> for $name<T>
			where T: tensor_macros::traits::TensorTrait + std::ops::MulAssign<U>,
			U: Clone,
		{
			fn mul_assign(&mut self, other: U) {
				for i in 0..mul!($($dim),*) {
					self[i] *= other.clone();
				}
			}
		}

		make_index_fn!($name; $($dim),*);
	};
}
