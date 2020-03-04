fn f(x: f64) -> f64 {
	let xx = x * x;
	let xxx = xx * x;
	let xxxx = xxx * x;
	1.0 + 4.0 * x + 6.0 * xx + 4.0 * xxx + xxxx
}

fn ternary_search(
	func: fn(f64) -> f64,
	left: f64,
	right: f64,
	eps: f64,
	max_iterations: i32) -> (f64, f64, i32) {

	let mut iterations = 0;
	let mut left = left;
	let mut right = right;

	while right - left > eps && iterations < max_iterations {
		let ml = left * 2.0 / 3.0 + right / 3.0;
		let mr = left / 3.0 + right * 2.0 / 3.0;

		if func(ml) < func(mr) {
			right = mr;
		} else {
			left = ml;
		}
		iterations += 1;
	}

	let r = left / 2.0 + right / 2.0;
	(r, func(r), iterations)
}

fn main() {
	let (min_at, min_val, iterations) = ternary_search(f, -2.0, 2.0, 1e-15, 1000);
	// let (min_at, min_val, iterations) = ternary_search(|x: f64| -> f64 {x * x}, -2.0, 2.0, 1e-15, 1000);
	println!("ternary_search iterations: {}", iterations);
	println!("min at: {}, value: {}", min_at, min_val);
}
