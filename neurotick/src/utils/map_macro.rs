#![allow(unused_macros)]
#![allow(unused_imports)]

use std::{vec, slice::Iter, collections::HashMap};

#[macro_export]
macro_rules! map {
    ($($k:expr => $v:expr),* $(,)?) => {{
        use std::iter::{Iterator, IntoIterator};
        Iterator::collect(IntoIterator::into_iter([$(($k, $v),)*]))
    }};
}
