#[macro_export]
/// Creates a tensor dot product function
///
/// # Example
///
/// ```rust

/// #![feature(try_from)]
///
/// #[macro_use]
/// use tensor_macros::*;
/// use tensor_macros::tensor::*;
///
/// tensor!(T243: 2 x 4 x 3);
/// tensor!(M43: 4 x 3 x 1);
/// tensor!(V2: 2 x 1);
///
/// dot!(T243: 2 x 4 x 3 * M43: 4 x 3 x 1 => V2: 2 x 1);
///
/// let l = T243([
///     0, 1, 2, 3,
///     4, 5, 6, 7,
///     
///     8, 9, 10, 11,
///     12, 13, 14, 15,
///
///     16, 17, 18, 19,
///     20, 21, 22, 23,
/// ]);
/// let r = M43([
///     0, 1, 2,
///     3, 4, 5,
///     6, 7, 8,
///     9, 10, 11
/// ]);
/// assert_eq!(l * r, V2([506, 1298]));
/// ```
macro_rules! dot {
    ($lhs:ident: $($l_dim:literal)x+ * $rhs:ident: $($r_dim:literal)x+ => $out:ident: $($o_dim:literal)x+) => {
        impl<T, U, V> std::ops::Mul<$rhs<U>> for $lhs<T>
        where
            T: tensor_macros::traits::TensorTrait + std::ops::Mul<U, Output=V>,
            U: tensor_macros::traits::TensorTrait,
            V: tensor_macros::traits::TensorTrait,
        {
            type Output = $out<V>;

            fn mul(self, rhs: $rhs<U>) -> Self::Output {
                let mut out = Self::Output::new();

                split!(self, rhs, out; $($l_dim),*; $($r_dim),*; $($o_dim),*;;;;;);

                out
            }
        }
    };
}

#[macro_export]
macro_rules! split {
    // left; right; output; right rev; count; out right; out left; out right rev
    // l    ; r    ; o      ; rr   ; c  ; or     ; ol; orr
    // 2 4 3; 3 2 1; 2 4 2 1;
    // 2 4 3;   2 1;     2 1; 3    ; 2  ; 2 4    ;
    // 2 4 3;     1;        ; 2 3  ; 2 2; 2 4 2 1;
    // 2 4 3;      ;        ; 1 2 3;   2;   4 2 1; 2
    // 2 4 3;      ;        ; 1 2 3;    ;     2 1; 2 4
    // 2 4 3;      ;        ; 1 2 3;    ;       1; 2 4; 2
    // 2 4 3;      ;        ; 1 2 3;    ;        ; 2 4; 1 2

    // 2 4 3; 4 3 1; 2 1;
    // 2 4 3;   3 1;    ;     4; 2; 2 1;  ;
    // 2 4 3;     1;    ;   3 4;  ;   1; 2;
    // 2 4 3;      ;    ; 1 3 4;  ;    ; 2; 1
    ($($i:expr),*;
        $($ls:literal),*; $r:literal $(,$rs:literal)*; $o1:literal, $o2:literal $(,$os:literal)*;
        $($rr:literal),*;
        $($c:literal),*; $($or:literal),*;;) => {
        split!($($i),*;
            $($ls),*; $($rs),*; $($os),*;
            $r $(,$rr)*;
            $($c,)* $o1; $($or,)* $o1, $o2;;
        );
    };

    ($($i:expr),*;
        $($ls:literal),*;; $o1:literal, $o2:literal $(,$os:literal)*;
        $($rr:literal),*;
        $($c:literal),*; $($or:literal),*;;) => {
        split!($($i),*;
            $($ls),*;; $($os),*;
            $($rr),*;
            $($c,)* $o1; $($or,)* $o1, $o2;;
        );
    };

    ($($i:expr),*;
        $($ls:literal),*; $r:literal $(,$rs:literal)*;;
        $($rr:literal),*;
        $c1:literal $(,$c:literal)*; $or1:literal $(,$or:literal)*; $($ol:literal),*;) => {
        split!($($i),*;
            $($ls),*; $($rs),*;;
            $r $(,$rr)*;
            $($c),*; $($or),*; $($ol,)* $or1;
        );
    };

    ($($i:expr),*;
        $($ls:literal),*;;;
        $($rr:literal),*;
        $c1:literal $(,$c:literal)*; $or1:literal $(,$or:literal)*; $($ol:literal),*;) => {
        split!($($i),*;
            $($ls),*;;;
            $($rr),*;
            $($c),*; $($or),*; $($ol,)* $or1;
        );
    };

    ($($i:expr),*;
        $($ls:literal),*; $r:literal $(,$rs:literal)*;;
        $($rr:literal),*;
        ; $or1:literal $(,$or:literal)*; $($ol:literal),*; $($orr:literal),*) => {
        split!($($i),*;
            $($ls),*; $($rs),*;;
            $r $(,$rr)*;
            ; $($or),*; $($ol),*; $or1 $(,$orr)*
        );
    };

    ($($i:expr),*;
        $($ls:literal),*;;;
        $($rr:literal),*;
        ; $or1:literal $(,$or:literal)*; $($ol:literal),*; $($orr:literal),*) => {
        split!($($i),*;
            $($ls),*;;;
            $($rr),*;
            ; $($or),*; $($ol),*; $or1 $(,$orr)*
        );
    };

    ($($i:expr),*;
        $($ls:literal),*; $r:literal $(,$rs:literal)*;;
        $($rr:literal),*;
        ;; $($ol:literal),*; $($orr:literal),*) => {
        split!($($i),*;
            $($ls),*; $($rs),*;;
            $r $(,$rr)*;
            ;; $($ol),*; $($orr),*
        );
    };

    ($($i:expr),*;
        $($ls:literal),*;;;
        $($rr:literal),*;
        ;; $($ol:literal),*; $($orr:literal),*) => {
        split!(~ $($i),*;
            $($ls),*;
            $($rr),*;
            $($ol),*; $($orr),*;;;
        );
    };

    // Actually performing the split
    // 2, 4, 3 * 3, 2, 1 -> 2, 4, 2, 1
    // 2, 4, 3 | 1, 2, 3 | 2, 4 | 1, 2 |      |      |
    //    4, 3 |    2, 3 |    4 |    2 | 2    |    1 |
    //       3 |       3 |      |      | 2, 4 | 2, 1 |   |
    //         |         |      |      | 2, 4 | 2, 1 | 3 |
    (~ $($i:expr),*;
        $l1:literal $(,$l:literal)*;
        $r1:literal $(,$r:literal)*;
        $ol1:literal $(,$ol:literal)*;
        $or1:literal $(,$or:literal)*;
        $($ld:literal),*; $($rd:literal),*;) => {
        assert_eq!($l1, $ol1, "Bad dimensions for tensor product");
        assert_eq!($r1, $or1, "Bad dimensions for tensor product");
        split!(~ $($i),*;
            $($l),*;
            $($r),*;
            $($ol),*; $($or),*;
            $($ld,)* $l1;
            $r1 $(,$rd)*;
        );
    };
    (~ $($i:expr),*;
        $l1:literal $(,$l:literal)*;
        $r1:literal $(,$r:literal),*;
        ;;
        $($ld:literal),*; $($rd:literal),*;
        $($sd:literal),*) => {
        split!(~ $($i),*;
            $($l),*;
            $($r),*;
            ;;
            $($ld),*; $($rd),*;
            $($sd,)* $l1
        );
    };
    (~ $($i:expr),*;
        ;;
        ;;
        $($ld:literal),*; $($rd:literal),*;
        $($sd:literal),*) => {
        make_dot!($($i),*; $($ld),*; $($sd),*; $($rd),*;;;);
    }
}

#[macro_export]
macro_rules! make_dot {
    ($l:expr, $r:expr, $o:expr; $d:literal $(,$ld:literal)*; $($sd:literal),*; $($rd:literal),*; $($lv:ident),*; $($sv:ident),*; $($rv:ident),*) => {
        for i in 0..$d {
            make_dot!($l, $r, $o; $($ld),*; $($sd),*; $($rd),*; $($lv,)* i; $($sv),*; $($rv),*);
        }
    };
    ($l:expr, $r:expr, $o:expr; ; $d:literal $(,$sd:literal)*; $($rd:literal),*; $($lv:ident),*; $($sv:ident),*; $($rv:ident),*) => {
        for j in 0..$d {
            make_dot!($l, $r, $o; ; $($sd),*; $($rd),*; $($lv),*; $($sv,)* j; $($rv),*);
        }
    };
    ($l:expr, $r:expr, $o:expr; ;; $d:literal $(,$rd:literal)*; $($lv:ident),*; $($sv:ident),*; $($rv:ident),*) => {
        for k in 0..$d {
            make_dot!($l, $r, $o; ;; $($rd),*; $($lv),*; $($sv),*; $($rv,)* k);
        }
    };
    ($l:expr, $r:expr, $o:expr; ;;; $($lv:ident),+; $($sv:ident),+; $($rv:ident),*) => {
        $o[($($lv),* $(,$rv),*)] += $l[($($lv),* $(,$sv)*)] * $r[($($sv,)* $($rv),*)]
    };
    ($l:expr, $r:expr, $o:expr; ;;;; $($sv:ident),+; $($rv:ident),+) => {
        $o[($($rv),*)] += $l[($($lv,)* $($sv)*)] * $r[($($sv),* $(,$rv)*)]
    };
    ($l:expr, $r:expr, $o:expr; ;;;; $($sv:ident),+;) => {
        $o += $l[($($lv),* $(,$sv)*)] * $r[($($sv,)* $($rv),*)]
    };
}

#[cfg(test)]
mod tests {
    use crate as tensor_macros;

    tensor!(T243: 2 x 4 x 3);
    tensor!(M43: 4 x 3 x 1);
    tensor!(V2: 2 x 1);

    dot!(T243: 2 x 4 x 3 * M43: 4 x 3 x 1 => V2: 2 x 1);

    #[test]
    fn dot() {
        let l = T243([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ]);
        let r = M43([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(l * r, V2([506, 1298]));
    }
}
