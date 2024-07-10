#![allow(dead_code)] #![allow(non_snake_case)] #![allow(unused_imports)]

use std::process::id;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{error, ops, vec};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Result, Write};

struct Random{
    state: u64,
}
impl Random {
    fn new() -> Self {
        let seed = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_nanos();
        Self {
            state: seed as u64
        }
    }
    fn set_seed(&mut self, seed: u64) {
        self.state = seed;
    }
    fn randint(&mut self, min: i32, max: i32) -> i32 {
        // random number generator in range [min, max]
        min + ((max+1 - min) as f32 * self.random()) as i32
    }
    fn update(&mut self){
        self.state = (self.state as u128 * 1103515245 as u128 + 12345 as u128) as u64 & 0x7fffffff;
    }
    fn random(&mut self) -> f32{
        // Use a simple linear congruential generator
        self.update();
        (self.state as f32)/(0x7fffffff as f32)
    }
}

struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    size: usize,
}
impl Tensor<f32> {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        Self { 
            data,
            shape, 
            size 
        }
    }
    fn ones(shape: Vec<usize>) -> Self {
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        let data: Vec<f32> = vec![1.0; size];
        Self { 
            data,
            shape, 
            size 
        }
    }
    fn zeros(shape: Vec<usize>) -> Self {
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        let data: Vec<f32> = vec![0.0; size];
        Self { 
            data,
            shape, 
            size 
        }
    }
    fn randn(shape: Vec<usize>, seed: Option<u64>) -> Self {
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        let mut random = Random::new();
        match seed {
            Some(g) => random.set_seed(g),
            None => (),
        }
        // TODO
        let mut data = Vec::<f32>::with_capacity(size);
        for _ in 0..size {
            data.push(random.random());
        }
        Self { 
            data,
            shape, 
            size 
        }
    }
    fn from(data: Vec<f32>, shape: Vec<usize>)->Self{
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        assert_eq!(data.len(), size);
        Self { data, shape, size }
    }
    fn add(&mut self, other: &Self){
        assert_eq!(self.shape, other.shape);
        for (idx, i) in other.data.iter().enumerate() {
            self.data[idx] += i;
        }
    }
    fn sub(&mut self, other: &Self){
        assert_eq!(self.shape, other.shape);
        for (idx, i) in other.data.iter().enumerate() {
            self.data[idx] -= i;
        }
    }
    fn mul(&mut self, other: &Self){
        if self.shape==other.shape {
            for idx in 0..self.size {
                self.data[idx]*=other.data[idx];
            }
        }else{
            assert!((self.shape.len() == 1 && self.shape[0] == 1)||(other.shape.len() == 1 && other.shape[0] == 1));
            if self.shape.len() == 1 && self.shape[0] == 1{
                let mut data = Vec::<f32>::with_capacity(other.size);
                for idx in 0..other.size {
                    data.push(other.data[idx]* self.data[0]);
                }
                self.data = data;
                self.size = other.size;
                self.shape = other.shape.clone();
            }else if other.shape.len() == 1 && other.shape[0] == 1 {
                for idx in 0..self.size {
                    self.data[idx]*=other.data[0];
                }
            }
        }
    }
    fn T(&mut self){
        // shape becomes [::-1]
        let mut shape = Vec::<usize>::with_capacity(self.shape.len());
        for i in (0..self.shape.len()).rev() {
            shape.push(self.shape[i]);
        }
        // let mut data = Vec::<f32>::with_capacity(self.size);
        let mut data = vec![0.0; self.size];

        for (idx, i) in self.data.iter().enumerate(){
            let mut new_idx: usize = 0;
            let mut idxmut = idx;
            for (jdx, j) in shape.iter().enumerate(){
                let mut multiplier = 1;
                for jj in 0..(shape.len()-1-jdx){
                    multiplier*=self.shape[jj];
                }
                new_idx += (idxmut % j)*multiplier;
                idxmut = (idxmut as f32 / *j as f32)as usize;
            }
            data[new_idx] = *i;
        }
        self.data = data;
        self.shape = shape;
    }
    fn reshape(&mut self, shape: Vec<usize>){
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        assert_eq!(self.size, size);
        self.shape = shape;
    }
    fn load(path: &str) -> Self {
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();

        let bytes = [
            buffer[0],
            buffer[1],
            buffer[2],
            buffer[3],
            buffer[4],
            buffer[5],
            buffer[6],
            buffer[7],
        ];
        let size = usize::from_ne_bytes(bytes);
        let mut data = Vec::<f32>::with_capacity(size);
        
        // Convert bytes to f32 values
        for i in 2..size+2 {
            let bytes = [
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
            ];
            let num = f32::from_ne_bytes(bytes);
            data.push(num);
        }
        // Calculate number of f32 values based on byte length
        let num_values = buffer.len() / std::mem::size_of::<f32>();
        
        let mut shape = Vec::<usize>::new();
        let mut i = size+2;
        while i<num_values {
            let bytes = [
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
                buffer[i * 4 + 4],
                buffer[i * 4 + 5],
                buffer[i * 4 + 6],
                buffer[i * 4 + 7],
            ];
            let num = usize::from_ne_bytes(bytes);
            shape.push(num);
            i+=2;
        }
        Self {
            data,
            shape,
            size
        }
    }
    fn save(&self, path: &str){
        let mut file = File::create(path).unwrap();
        file.write_all(&self.size.to_ne_bytes()).unwrap();
        for &num in self.data.iter() {
            file.write_all(&num.to_ne_bytes()).unwrap();
        }
        for &num in self.shape.iter() {
            file.write_all(&num.to_ne_bytes()).unwrap();
        }
    }
}

trait PrintHelper {
    fn print_helper(&self, index: usize, i: usize)->usize;
}
impl PrintHelper for Tensor<f32> {
    fn print_helper(&self, index: usize, i: usize)->usize{
        let mut index = index;
        let threshold: usize = 7;
        if i >= self.shape.len()-1 {
            print!("[");
            if self.shape[i]-1>=threshold{
                for _ in 0..3 {
                    print!("{:.4}, ", self.data[index]);
                    index+=1;
                }
                index+=self.shape[i]-6;
                print!("..., ");
                for _ in 0..2 {
                    print!("{:.4}, ", self.data[index]);
                    index+=1;
                }
            }else{
                for _ in 0..self.shape[i]-1 {
                    print!("{:.4}, ", self.data[index]);
                    index+=1;
                }
            }

            print!("{:.4}]", self.data[index]);
            index+=1;
            return index;
        }else{
            print!("[");
            if self.shape[i]-1>=threshold{
                for _ in 0..3 {
                    index = self.print_helper(index, i+1);
                    print!(",");
                    for _ in 0..self.shape.len()-i-1 {print!("\n");}
                    for _ in 0..i+1+7 {print!(" ");}
                }
                let mut t: usize=1;
                for j in (i+1)..self.shape.len(){
                    t=t*self.shape[j];
                }
                index+=t*(self.shape[i]-6);
                print!("...,");
                for _ in 0..self.shape.len()-i-1 {print!("\n");}
                for _ in 0..i+1+7 {print!(" ");}
                for _ in 0..2 {
                    index = self.print_helper(index, i+1);
                    print!(",");
                    for _ in 0..self.shape.len()-i-1 {print!("\n");}
                    for _ in 0..i+1+7 {print!(" ");}
                }
            }else{
                for _ in 0..self.shape[i]-1 {
                    index = self.print_helper(index, i+1);
                    print!(",");
                    for _ in 0..self.shape.len()-i-1 {print!("\n");}
                    for _ in 0..i+1+7 {print!(" ");}
                }
            }
            index = self.print_helper(index, i+1);
            print!("]");
            return index;
        }
    }
}
impl std::fmt::Display for Tensor<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        print!("Tensor(");
        // int index = 0;
        self.print_helper(0, 0);
        print!(", shape=(");
        for i in 0..self.shape.len()-1 {
            print!("{}, ", self.shape[i]);
        }
        print!("{})", self.shape[self.shape.len()-1]);
        write!(f, ", size={}, dtype=f32)", self.size)
    }
}

// TODO broadcasting... (done => (Add 10%, Mul 10%, Sub 10%, ))
impl ops::Add for Tensor<f32>{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output{
        if self.shape==rhs.shape {
            let mut data = Vec::<f32>::with_capacity(self.size);
            for (idx, i) in self.data.iter().enumerate() {
                data.push(i+rhs.data[idx]);
            }
            return Self::new(data, self.shape);
        }else{
            assert!((self.shape.len() == 1 && self.shape[0] == 1)||(rhs.shape.len() == 1 && rhs.shape[0] == 1));
            if self.shape.len() == 1 && self.shape[0] == 1{
                return rhs + self.data[0];
            }else if rhs.shape.len() == 1 && rhs.shape[0] == 1 {
                return self + rhs.data[0];
            }else{
                todo!()
            }
        } 
    }
}
impl ops::Add<f32> for Tensor<f32> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        let mut data = Vec::<f32>::with_capacity(self.size);
        for i in self.data.iter() {
            data.push(i+rhs);
        }
        Self::new(data, self.shape)
    }
}
impl ops::Add<Tensor<f32>> for f32 {
    type Output = Tensor<f32>;

    fn add(self, rhs: Tensor<f32>) -> Self::Output {
        rhs + self
    }
}
impl ops::Sub for Tensor<f32>{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape==rhs.shape {
            let mut data = Vec::<f32>::with_capacity(self.size);
            for (idx, i) in self.data.iter().enumerate() {
                data.push(i-rhs.data[idx]);
            }
            return  Self::new(data, self.shape);
        }else{
            assert!((self.shape.len() == 1 && self.shape[0] == 1)||(rhs.shape.len() == 1 && rhs.shape[0] == 1));
            if self.shape.len() == 1 && self.shape[0] == 1{
                return rhs - self.data[0];
            }else if rhs.shape.len() == 1 && rhs.shape[0] == 1 {
                return self - rhs.data[0];
            }else{
                todo!()
            }
        }
    }
}
impl ops::Sub<f32> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, rhs: f32) -> Self::Output {
        let mut data = Vec::<f32>::with_capacity(self.size);
        for i in self.data.iter() {
            data.push(i-rhs);
        }
        Self::new(data, self.shape)
    }
}
impl ops::Sub<Tensor<f32>> for f32 {
    type Output = Tensor<f32>;

    fn sub(self, rhs: Tensor<f32>) -> Self::Output {
        let mut data = Vec::<f32>::with_capacity(rhs.size);
        for i in rhs.data.iter() {
            data.push(self-i);
        }
        Self::Output::new(data, rhs.shape)
    }
}
impl ops::AddAssign for Tensor<f32>{
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.size, rhs.size);
    
        for (idx, i) in rhs.data.iter().enumerate() {
            self.data[idx] += i;
        }
    }
}
impl ops::SubAssign for Tensor<f32> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.size, rhs.size);
    
        for (idx, i) in rhs.data.iter().enumerate() {
            self.data[idx] -= i;
        }
    }
}
impl ops::AddAssign<f32> for Tensor<f32> {
    fn add_assign(&mut self, rhs: f32) {
        for idx in 0..self.size {
            self.data[idx] += rhs;
        }
    }
}
impl ops::SubAssign<f32> for Tensor<f32> {
    fn sub_assign(&mut self, rhs: f32) {
        for idx in 0..self.size {
            self.data[idx] -= rhs;
        }
    }
}
impl ops::MulAssign<f32> for Tensor<f32> {
    fn mul_assign(&mut self, rhs: f32) {
        for idx in 0..self.size {
            self.data[idx] *= rhs;
        }
    }
}
impl ops::DivAssign<f32> for Tensor<f32> {
    fn div_assign(&mut self, rhs: f32) {
        for idx in 0..self.size {
            self.data[idx] /= rhs;
        }
    }
}
impl ops::MulAssign for Tensor<f32> {
    fn mul_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        for (idx, i) in rhs.data.iter().enumerate() {
            self.data[idx] *= i;
        }
    }
}
impl ops::DivAssign for Tensor<f32> {
    fn div_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.size, rhs.size);
    
        for (idx, i) in rhs.data.iter().enumerate() {
            self.data[idx] /= i;
        }
    }
}
impl ops::Mul for Tensor<f32> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape==rhs.shape {
            let mut data = Vec::<f32>::with_capacity(self.size);
            for (idx, i) in self.data.iter().enumerate() {
                data.push(i*rhs.data[idx]);
            }
            return Self::new(data, self.shape);
        }else{
            assert!((self.shape.len() == 1 && self.shape[0] == 1)||(rhs.shape.len() == 1 && rhs.shape[0] == 1));
            if self.shape.len() == 1 && self.shape[0] == 1{
                return rhs * self.data[0];
            }else if rhs.shape.len() == 1 && rhs.shape[0] == 1 {
                return self * rhs.data[0];
            }else{
                todo!()
            }
        }
    }
}
impl ops::Div for Tensor<f32> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(self.size, rhs.size);
        let mut data = Vec::<f32>::with_capacity(self.size);
        for (idx, i) in self.data.iter().enumerate() {
            data.push(i/rhs.data[idx]);
        }
        Self::new(data, self.shape)
    }
}
impl ops::Mul<f32> for Tensor<f32> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut data = Vec::<f32>::with_capacity(self.size);
        for i in self.data.iter() {
            data.push(i*rhs);
        }
        Self::new(data, self.shape)
    }
}
impl ops::Div<f32> for Tensor<f32> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut data = Vec::<f32>::with_capacity(self.size);
        for i in self.data.iter() {
            data.push(i/rhs);
        }
        Self::new(data, self.shape)
    }
}
impl ops::Mul<Tensor<f32>> for f32 {
    type Output = Tensor<f32>;
    fn mul(self, rhs: Tensor<f32>) -> Self::Output {
        rhs * self
    }
}
impl ops::Div<Tensor<f32>> for f32 {
    type Output = Tensor<f32>;
    fn div(self, rhs: Tensor<f32>) -> Self::Output {
        let mut data = Vec::<f32>::with_capacity(rhs.size);
        for i in rhs.data.iter() {
            data.push(self/i);
        }
        Self::Output::new(data, rhs.shape)
    }
}
impl ops::Neg for Tensor<f32> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * (-1.0)
    }
}
impl ops::Mul<Matmul> for Tensor<f32> {
    type Output = Self;

    fn mul(self, other: Matmul) -> Self::Output {
        let Matmul::Data(rhs) = other;
        assert_eq!(self.shape.len(), rhs.shape.len());
        assert_eq!(self.shape[self.shape.len() - 1], rhs.shape[rhs.shape.len() - 2]);
        assert_eq!(&self.shape[..self.shape.len() - 2], &rhs.shape[..rhs.shape.len() - 2]);
        let mut shape = Vec::<usize>::with_capacity(self.shape.len());
        for i in &self.shape {
            shape.push(*i);
        }
        shape[self.shape.len() - 1] = rhs.shape[rhs.shape.len() - 1];
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        
        let mut data = Vec::<f32>::with_capacity(size);
        
        Matmul::helper(&mut data, &self, &rhs, 0, 0, 0);

        Self::new(data, shape)
    }
}
impl ops::MulAssign<Matmul> for Tensor<f32> {
    fn mul_assign(&mut self, other: Matmul) {
        let Matmul::Data(rhs) = other;
        assert_eq!(self.shape.len(), rhs.shape.len());
        assert_eq!(self.shape[self.shape.len() - 1], rhs.shape[rhs.shape.len() - 2]);
        assert_eq!(&self.shape[..self.shape.len() - 2], &rhs.shape[..rhs.shape.len() - 2]);
        let mut shape = Vec::<usize>::with_capacity(self.shape.len());
        for i in &self.shape {
            shape.push(*i);
        }
        shape[self.shape.len() - 1] = rhs.shape[rhs.shape.len() - 1];
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        let mut data = Vec::<f32>::with_capacity(size);
        
        Matmul::helper(&mut data, &self, &rhs, 0, 0, 0);

        self.data = data;
        self.shape = shape;
        self.size = size;
    }
}

enum Matmul{
    Data(Tensor<f32>)
}
impl Matmul {
    fn helper(data: &mut Vec<f32>, lhs: &Tensor<f32>, rhs: &Tensor<f32>, index_rhs: usize,  index_lhs: usize, d: usize)->(usize, usize){
        let mut index_rhs = index_rhs;
        let mut index_lhs = index_lhs;
        if lhs.shape.len()-2 > d{
            for _ in 0..lhs.shape[d] {
                (index_rhs, index_lhs) = Matmul::helper(data, lhs, rhs, index_rhs, index_lhs, d+1);    
            }
        }else{
            for i in 0..lhs.shape[lhs.shape.len() - 2] {
                for j in 0..rhs.shape[rhs.shape.len() - 1] {
                    let mut sum = 0.0;
                    for k in 0..lhs.shape[lhs.shape.len() - 1] {
                        sum += lhs.data[index_lhs + i * lhs.shape[lhs.shape.len() - 1] + k] * rhs.data[index_rhs + k * rhs.shape[rhs.shape.len() - 1] + j];
                    }
                    data.push(sum);
                }
            }
            index_rhs += rhs.shape[lhs.shape.len() - 2] * rhs.shape[rhs.shape.len() - 1];
            index_lhs += lhs.shape[lhs.shape.len() - 2] * lhs.shape[rhs.shape.len() - 1];
        }
        (index_rhs, index_lhs)
    }
}
impl ops::Not for Tensor<f32> {
    type Output = Matmul;
    fn not(self) -> Self::Output {
        Self::Output::Data(self)
    }
}

struct Flash {
    seed: Option<u64>,
}
impl Flash {
    fn new() -> Self {
        Self { seed: None }
    }
    fn linalg_solve(a: &Tensor<f32>, b: &Tensor<f32>)->Tensor<f32>{
        // as of now, implementation for 1 D array only
        assert!(a.shape.len()==2 && a.shape[0]==a.shape[1]);
        assert!(b.shape.len()==1 && a.shape[0]==b.shape[0]);
        let n = a.shape[0];
        let m = a.shape[1];
        let mut augmented = Flash::column_stack(a, b);
        // println!("{}", augmented);
        for i in 0..n{
            let mut maxidx: usize = 0;
            let mut maxval: f32 = 0.0;
            for j in i..m{
                let mut val = augmented.data[j*m+j+i];
                if val<0.0{
                    val*=-1.0;
                }
                if val>maxval{
                    maxval=val;
                    maxidx = j-i;
                }
            }
            
            // Find pivot
            let max_element = maxidx+i;

            if i != max_element{
                // augmented[[i, max_element]] = augmented[[max_element, i]]
                // swap i col with max_element col
                let mut temp = Vec::<f32>::with_capacity(m+1);
                for j in 0..=m{
                    temp.push(augmented.data[i*m+i+j]);
                }
                
                for j in 0..=m{
                    augmented.data[i*m+i+j] = augmented.data[max_element*m+max_element+j];
                }
                for j in 0..=m{
                    augmented.data[max_element*m+max_element+j] = temp[j];
                }
            }
            
            // Check for singular matrix
            assert!(augmented.data[i*m+i+i] != 0.0);

            // Eliminate below
            for j in (i+1)..n{
                let factor = augmented.data[j*m+j+i]/augmented.data[i*m+i+i];
                // augmented[j, i:] -= factor * augmented[i, i:]
                for idx in i..=m{
                    augmented.data[j*m+j+idx] -= factor*augmented.data[i*m+i+idx];
                }
            }
        }
        // Back substitution
        let mut x = Tensor::zeros(vec![n]);
        for i in (0..n).rev(){
            // res = np.dot(augmented[i, i+1:n], x[i+1:])
            let mut res = 0.0;
            for j in (i+1)..n{
                res += augmented.data[i*m+i+j]*x.data[j];
            }
            // x[i] = (augmented[i, -1] - res) / augmented[i, i]
            x.data[i] = (augmented.data[i*m+i+m] - res)/augmented.data[i*m+i+i];
        }
        x
    }
    fn abs(a: &Tensor<f32>) -> Tensor<f32> {
        let mut data = Vec::<f32>::with_capacity(a.size);
        for i in a.data.iter() {
            if (*i)<0.0{
                data.push(-(*i));
            }else{
                data.push(*i);
            }
        }
        Tensor::new(data, a.shape.clone())
    }
    fn column_stack(a: &Tensor<f32>, b: &Tensor<f32>)->Tensor<f32>{
        // let shape
        if a.shape.len() == 1 && b.shape.len() == 1 && a.shape[0] == b.shape[0]{
            let shape = vec![a.size, a.size];
            let mut data = Vec::<f32>::with_capacity(a.size*a.size);
            for (idx, i) in a.data.iter().enumerate() {
                data.push(*i);
                data.push(b.data[idx]);
            }
            Tensor::new(data, shape)
        }else if a.shape.len() == 2 && b.shape.len() == 1 && a.shape[0] == b.shape[0]{
            let shape = vec![a.shape[0], a.shape[1]+1];
            let mut data = Vec::<f32>::with_capacity(a.size+b.size);
            for (idx, chunk) in a.data.chunks(a.shape[1]).enumerate() {
                for i in chunk.iter(){
                    data.push(*i);
                }
                data.push(b.data[idx]);
            }
            Tensor::new(data, shape)
        }else if a.shape.len() == 1 && b.shape.len() == 2 && a.shape[0] == b.shape[0]{
            // TODO use broadcasting a:(2) b:(2, 3) make a => a(2, 1)
            todo!()
        }else if a.shape.len() == b.shape.len(){
            for idx in 0..a.shape.len(){
                if idx == 1{ continue; }
                assert_eq!(a.shape[idx], b.shape[idx]);
            }
            todo!()
        }else{
            todo!()
        }
    }
    fn outer(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32>{
        // flatten if not flatten
        let shape: Vec<usize> = vec![a.size, b.size];
        let mut data = Vec::<f32>::with_capacity(a.size*b.size);
        for i in a.data.iter() {
            for j in b.data.iter() {
                data.push(i*j);
            }
        }
        Tensor::new(data, shape)
    }
    fn T(a: &Tensor<f32>) -> Tensor<f32> {
        // shape becomes [::-1]
        let mut shape = Vec::<usize>::with_capacity(a.shape.len());
        for i in (0..a.shape.len()).rev() {
            shape.push(a.shape[i]);
        }
        // let mut data = Vec::<f32>::with_capacity(self.size);
        let mut data = vec![0.0; a.size];

        for (idx, i) in a.data.iter().enumerate(){
            let mut new_idx: usize = 0;
            let mut idxmut = idx;
            for (jdx, j) in shape.iter().enumerate(){
                let mut multiplier = 1;
                for jj in 0..(shape.len()-1-jdx){
                    multiplier*=a.shape[jj];
                }
                new_idx += (idxmut % j)*multiplier;
                idxmut = (idxmut as f32 / *j as f32)as usize;
            }
            data[new_idx] = *i;
        }
        Tensor::new(data, shape)
    }
    fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32>{
        assert_eq!(a.shape.len(), b.shape.len());
        assert_eq!(a.shape[a.shape.len() - 1], b.shape[b.shape.len() - 2]);
        assert_eq!(&a.shape[..a.shape.len() - 2], &b.shape[..b.shape.len() - 2]);
        let mut shape = Vec::<usize>::with_capacity(a.shape.len());
        for i in &a.shape {
            shape.push(*i);
        }
        shape[a.shape.len() - 1] = b.shape[b.shape.len() - 1];
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        
        let mut data = Vec::<f32>::with_capacity(size);
        
        Matmul::helper(&mut data, &a, &b, 0, 0, 0);

        Tensor::new(data, shape)
    }
    fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32>{
        if a.shape==b.shape {
            let mut data = Vec::<f32>::with_capacity(a.size);
            for (idx, i) in a.data.iter().enumerate() {
                data.push(i+b.data[idx]);
            }
            return Tensor::new(data, a.shape.clone())
        }else{
            assert!((a.shape.len() == 1 && a.shape[0] == 1)||(b.shape.len() == 1 && b.shape[0] == 1));
            if a.shape.len() == 1 && a.shape[0] == 1{
                let mut data = Vec::<f32>::with_capacity(b.size);
                for i in b.data.iter() {
                    data.push(a.data[0]+(*i));
                }
                return Tensor::new(data, b.shape.clone())
            }else if b.shape.len() == 1 && b.shape[0] == 1 {
                let mut data = Vec::<f32>::with_capacity(a.size);
                for i in a.data.iter() {
                    data.push(b.data[0]+(*i));
                }
                return Tensor::new(data, a.shape.clone())
            }else{
                todo!()
            }
        }
    }
    fn sub(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32>{
        let mut data = Vec::<f32>::with_capacity(a.size);
        for (idx, i) in a.data.iter().enumerate() {
            data.push(i-b.data[idx]);
        }
        Tensor::new(data, a.shape.clone())
    }
    fn copy(a: &Tensor<f32>)->Tensor<f32>{
        let mut data = Vec::<f32>::with_capacity(a.size);
        for i in a.data.iter() {
            data.push(*i);
        }
        Tensor::new(data, a.shape.clone())
    }
    // TODO check this implementation as i just copied from mean ...
    fn sum(a: &Tensor<f32>, dim: Option<usize>) -> Tensor<f32>{
            let axis: usize;
            if let Some(dim) = dim {
                axis = dim;
            }else{
                let mut sum: f32 = 0.0;
                for i in a.data.iter() {
                    sum += i;
                }
                return Tensor::new(vec![sum], vec![1]);
            }
            assert!(axis<a.shape.len());
            let mut shape = Vec::<usize>::with_capacity(a.shape.len()-1);
            for i in 0..a.shape.len() {
                if i == axis{ continue; }
                shape.push(a.shape[i]);
            }
            let mut size: usize = 1;
            for i in shape.iter() {
                size *= i;
            }
            // println!("{:?}", shape);
    
            let mut jump: usize = 1;
            for i in axis..shape.len(){
                jump *= shape[i];
            }
            
            let mut data = vec![0.0; size];
            
            for (idx, chunk) in a.data.chunks(jump).enumerate() {
                let d_index = (idx/a.shape[axis])*jump;
                for (j, i) in chunk.iter().enumerate(){
                    data[d_index+j] += i;
                }
            }
            Tensor::new(data, shape)
    }
    fn mean(a: &Tensor<f32>, dim: Option<usize>) -> Tensor<f32>{
        let axis: usize;
        if let Some(dim) = dim {
            axis = dim;
        }else{
            let mut sum: f32 = 0.0;
            for i in a.data.iter() {
                sum += i;
            }
            return Tensor::new(vec![sum/a.size as f32], vec![1]);
        }
        assert!(axis<a.shape.len());
        let mut shape = Vec::<usize>::with_capacity(a.shape.len()-1);
        for i in 0..a.shape.len() {
            if i == axis{ continue; }
            shape.push(a.shape[i]);
        }
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        // println!("{:?}", shape);

        let mut jump: usize = 1;
        for i in axis..shape.len(){
            jump *= shape[i];
        }
        
        let mut data = vec![0.0; size];
        
        for (idx, chunk) in a.data.chunks(jump).enumerate() {
            let d_index = (idx/a.shape[axis])*jump;
            for (j, i) in chunk.iter().enumerate(){
                data[d_index+j] += i;
            }
        }
        for i in 0..size{
            data[i] /= a.shape[axis] as f32;
        }
        Tensor::new(data, shape)
    }
    fn reshape(a: &Tensor<f32>, shape:  Vec<usize>)->Tensor<f32>{
        let mut size: usize = 1;
        for i in shape.iter() {
            size *= i;
        }
        assert_eq!(a.size, size);
        let mut data = Vec::<f32>::with_capacity(size);
        for i in a.data.iter() {
            data.push(*i);
        }
        Tensor::new(data, shape)
    }
    fn repeat(a: &Tensor<f32>, repeats:usize, dim: Option<usize>) -> Tensor<f32>{
        let axis: usize;
        if let Some(dim) = dim {
            axis = dim;
        }else{
            let mut data = Vec::<f32>::with_capacity(a.size*repeats);
            for i in a.data.iter() {
                for _ in 0..repeats{
                    data.push(*i);
                }
            }
            return Tensor::new(data, vec![a.size*repeats]);
        }
        let mut shape = Vec::<usize>::with_capacity(a.shape.len());
        for i in 0..a.shape.len() {
            if i == axis{ 
                shape.push(a.shape[i]*repeats);
             }
            else{
                shape.push(a.shape[i]);
            }
        }
        
        let mut jump: usize = 1;
        for i in axis+1..shape.len(){
            jump *= shape[i];
        }
        
        
        let mut data = Vec::<f32>::with_capacity(a.size*repeats);
        for chunk in a.data.chunks(jump) {
            for _ in 0..repeats{
                for i in chunk{
                    data.push(*i);
                }
            }
        }

        Tensor::new(data, shape)
    }
}

fn fit_linear_regression_gradient_descent(x: &Tensor<f32>, y: &Tensor<f32>, epochs: usize, lr: f32)->(Tensor<f32>, Tensor<f32>){
    let mut m = Tensor::randn(vec![x.shape[1], 1], None);
    let mut b = Tensor::randn(vec![1], None);
    for epoch in 1..=epochs {
        let mut errors = Flash::sub(&y, 
                                             &Flash::add(&Flash::matmul(&x, &m), 
                                                              &b));
        errors.reshape(vec![1, errors.size]);
        let mut d_ed_m = Flash::matmul(&errors, &x);
        d_ed_m.reshape(vec![d_ed_m.size, 1]);
        d_ed_m = -2.0*d_ed_m*lr;
        m = m - d_ed_m;

        let d_ed_b = -2.0*Flash::mean(&errors, None)*lr;
        b = b - d_ed_b;
        if epoch % 1 == 0 {
            println!("[{}/{}] Loss: {}", epoch, epochs, Flash::mean(&errors, None));
        }
    }
    return (m, b);
}
fn fit_linear_regression(x: &Tensor<f32>, y: &Tensor<f32>)->(Tensor<f32>, Tensor<f32>){
    let n = x.shape[0];
    let mut avg_y = Flash::mean(y, None);
    let mut avg_xj = Flash::mean(x, Some(0));
    
    let mut newshapey = y.shape.clone(); newshapey.push(1);
    let mut avg_yxj = Flash::repeat(
        &Flash::reshape(y, newshapey),
        x.shape[1], Some(1)
    );
    avg_yxj.mul(&x);
    avg_yxj = Flash::mean(&avg_yxj, Some(0));

    let avg_xjxj = Flash::matmul(
        &Flash::T(x),
        x
    ) / n as f32;

    let mut b = Flash::copy(&avg_y);

    let A = Flash::outer(&avg_xj, &avg_xj) - avg_xjxj;
    avg_y.mul(&avg_xj);
    let B = avg_y - avg_yxj;
    
    let mut M = Flash::linalg_solve(&A, &B);
    
    // let b = np.dot(M, avg_xj)
    avg_xj.mul(&M);
    b -= Flash::sum(&avg_xj, None);

    // println!("{}", M);
    // println!("{}", b);
    M.reshape(vec![M.size, 1]);
    (M, b)
}

fn main(){
    let X_train: Tensor<f32> = Tensor::load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\X_train.bin");
    let Y_train = Tensor::load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\Y_train.bin");
    let X_test = Tensor::load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\X_test.bin");
    println!("File Loaded");
    // let X  = Tensor::load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\X.bin");
    // let Y  = Tensor::load(r"C:\ThefCraft\thefcraft-rust\nn-c\src\Y.bin");
    // let (m, b) = fit_linear_regression_gradient_descent(&X_train, &Y_train, 10, 0.0003);
    let (m, b) = fit_linear_regression(&X_train, &Y_train);
    let model = |a: &Tensor<f32>| Flash::add(&Flash::matmul(&a, &m), &b);
    
    model(&X_test).save(r"C:\ThefCraft\thefcraft-rust\nn-c\src\y_pred.bin");
    // model(&X_train);
    // println!("{}", y);
}
