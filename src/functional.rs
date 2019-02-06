// This file is pretty useless
// I originally made it with the intention that
// rust macros could be used as arguments and expand early
// That is not the case yet
// There are ideas to make this a reality
// but for now, I cannot use these macros

#[macro_export]
/// Generate consecutive pairs from a list of inputs
///
/// Used internally by other macros.
/// Takes the first two values,
/// adds them both to the end and discards the first one
///
/// # Example
///
/// ```rust
/// use tensor_macros::pairs;
/// let v = pairs!(1, 2, 3, 4, 5);
/// assert_eq!(v, ((1, 2), (2, 3), (3, 4), (4, 5)));
/// ```
macro_rules! pairs {
    ($_1:literal, $_2:literal $(,$tail:literal)*) => {
        pairs!($_2 $(,$tail)*; $_1, $_2);
    };

    ($_1:literal, $_2:literal $(,$tail:literal)*; $($x:literal, $y:literal),*) => {
        pairs!($_2 $(,$tail)*; $($x, $y),*, $_1, $_2);
    };

    ($_1:literal; $($x:literal, $y:literal),*) => {
        ($(($x, $y)),*)
    };
}

#[macro_export]
macro_rules! rev {
    // 1 2 3 4;
    //   2 3 4; 1
    //     3 4; 2 1
    //       4; 3 2 1
    //        ; 4 3 2 1
    ($z:literal $(,$x:literal)*; $($y:literal),*) => (
        rev!($($x),*; $z $(,$y)*);
    );
    (; $($y:literal),*) => (
        ($($y),*)
    );
}

macro_rules! pad {
    (;;$($l:literal),*; $($r:literal),*; $($o:literal),*) => {
        (($($l),*),($($r),*), ($($o),*))
    };
    ($x:literal $(,$y:literal)*; ; $($l:literal),*; $($r:literal),*; $($o:literal),*) => {
        pad!($($y),*; ; $($l,)* $x; $($r,)* 1; $($o,)* 1);
    };
    (; $x:literal $(,$y:literal)*; $($l:literal),*; $($r:literal),*; $($o:literal),*) => {
        pad!(; $($y),*; 1 $(,$l)*; $($r,)* $x; 1 $(,$o)*);
    };
    ($x1:literal $(,$y1:literal)*; $x2:literal $(,$y2:literal)*; $($l:literal),*; $($r:literal),*; $($o:literal),*) => {
        pad!($($y1),*; $($y2),*; $($l,)* $x1; $($r,)* $x2; $($o),*);
    };
}

macro_rules! half {
    // 1 2 3 4;     ;        ;
    //     3 4; 1   ; 1 2    ;
    //        ; 1 3 ; 1 2 3 4;
    ($x:literal, $y:literal $(,$z:literal)*; $($c:literal),*; $($b:literal),*;) => (
        half!($($z),*; $($c,)* $x; $($b,)* $x, $y;);
    );
    //        ;   3 ;   2 3 4; 1
    //        ;     ;     3 4; 1 2
    (; $c1:literal $(,$c:literal)*; $b1:literal $(,$b:literal)*; $($a:literal),*) => (
        half!(; $($c),*; $($b),*; $($a,)* $b1);
    );
    (;; $($b:literal),*; $($a:literal),*) => {
        (($($a),*), ($($b),*))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pairs() {
        let v = pairs!(1, 2, 3, 4, 5);
        assert_eq!(v, ((1, 2), (2, 3), (3, 4), (4, 5)));
    }

    #[test]
    fn rev() {
        assert_eq!(rev!(1, 2, 3, 4;), (4, 3, 2, 1));
    }

    #[test]
    fn pad() {
        assert_eq!(
            pad!(2, 4, 3; 3, 2;;; 2, 4, 2),
            ((2, 4, 3), (3, 2, 1), (2, 4, 2, 1))
        );
        assert_eq!(
            pad!(2, 4, 3; 3;;; 2, 4),
            ((2, 4, 3), (3, 1, 1), (2, 4, 1, 1))
        );
        assert_eq!(
            pad!(2, 4; 4, 2, 3;;; 2, 2, 3),
            ((1, 2, 4), (4, 2, 3), (1, 2, 2, 3))
        );
    }

    #[test]
    fn half() {
        assert_eq!(
            half!(0, 1, 2, 3, 4, 5, 6, 7;;;),
            ((0, 1, 2, 3), (4, 5, 6, 7))
        );
    }
}
