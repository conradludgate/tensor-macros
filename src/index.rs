#[macro_export]
macro_rules! make_index_fn {
	($name:ident; $dim:literal) => {
		impl<T> std::ops::Index<usize> for $name<T> {
			type Output = T;

			fn index(&self, i: usize) -> &Self::Output {
				&self.0[i]
			}
		}

		impl<T> std::ops::IndexMut<usize> for $name<T> {
			fn index_mut(&mut self, i: usize) -> &mut T {
				&mut self.0[i]
			}
		}
	};

	($name:ident; $($dims:literal),*) => {
		impl<T> std::ops::Index<usize> for $name<T> {
			type Output = T;

			fn index(&self, i: usize) -> &Self::Output {
				&self.0[i]
			}
		}

		impl<T> std::ops::IndexMut<usize> for $name<T> {
			fn index_mut(&mut self, i: usize) -> &mut T {
				&mut self.0[i]
			}
		}

		make_index_fn!($name; $($dims),*;;;);
	};

	($name:ident; $dim:literal $(,$dims:literal)*; $($i:ident),*; $($t:ty),*; $($dims_bk:literal),*) => {
		make_index_fn!($name; $($dims),*; $($i,)* i; $($t,)* usize; $($dims_bk,)* $dim);
	};
	($name:ident; ; $($i:ident),*; $($t:ty),*; $($dims:literal),*) => {
		impl<T> std::ops::Index<( $($t),* )> for $name<T> {
			type Output = T;

			fn index(&self, ( $($i),* ): ( $($t),* )) -> &Self::Output {
				&self.0[
					make_index_val!($($dims),*; $($i),*)
				]
			}
		}

		impl<T> std::ops::IndexMut<( $($t),* )> for $name<T> {
			fn index_mut(&mut self, ( $($i),* ): ( $($t),* )) -> &mut T {
				&mut self.0[
					make_index_val!($($dims),*; $($i),*)
				]
			}
		}
	};
}

#[macro_export]
macro_rules! make_index_val {
	($dim:literal $(,$dims:literal)*; $i:expr $(,$is:expr)* ) => (
		$i * mul!($($dims),*) + make_index_val!($($dims),*; $($is),*)
	);
	(;) => (0)
}
