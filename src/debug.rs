#[macro_export]
macro_rules! debug_tensor {
	($w:ident, $t:ident; $d:literal $(,$ds:literal)+; $($is:ident),*) => {
		for i in 0..$d {
			debug_tensor!($w, $t; $($ds),*; $($is,)* i);
			writeln!($w)?;
		}
	};
	($w:ident, $t:ident; $d:literal; $($is:ident),+) => {
		for i in 0..$d {
			write!($w, "{:?}\t", $t[($($is,)* i)])?;
		}
	};
	($w:ident, $t:ident; $d:literal; ($i:ident)) => {
		for i in 0..$d {
			write!($w, "{:?}\t", $t[($i, i)])?;
		}
	};
	($w:ident, $t:ident; $d:literal;) => {
		for i in 0..$d {
			write!($w, "{:?}\t", $t[i])?;
		}
	};
}
