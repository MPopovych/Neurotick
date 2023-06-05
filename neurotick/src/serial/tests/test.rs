#[cfg(test)]
mod test {
    use crate::matrix::nmatrix::NDMatrix;

    #[test]
    fn serialize_weight() {
        let pre = NDMatrix::constant(3, 4, 1.5);
        dbg!(&pre);
        let json = serde_json::to_string(&pre).unwrap();
        dbg!(&json);
        let post: NDMatrix = serde_json::from_str(&json).unwrap();
        dbg!(&post);

        assert_eq!(&pre.values, &post.values)
    }
}
