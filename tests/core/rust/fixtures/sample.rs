use std::fmt;

/// Sample struct
pub struct User {
    pub id: u32,
    pub name: String,
}

impl User {
    pub fn new(id: u32, name: String) -> Self {
        User { id, name }
    }

    pub fn display(&self) -> String {
        format!("{}:{}", self.id, self.name)
    }
}

pub fn make_user() -> User {
    User::new(1, "Ada".to_string())
}

pub trait Printable {
    fn print(&self) -> String;
}

impl Printable for User {
    fn print(&self) -> String {
        self.display()
    }
}
