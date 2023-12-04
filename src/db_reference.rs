use rocksdb::DB;
use std::cell::RefCell;
use std::sync::Arc;

/// The type of a reference to a [rocksdb::DB] that is passed around the library.
pub(crate) type DbReference = Arc<RefCell<DB>>;

/// A wrapper around [DbReference] that cancels all background work when dropped.
///
/// All users of [rocksdb::DB] should use this wrapper instead to avoid keeping background threads
/// alive after the database is dropped.
#[derive(Clone)]
pub(crate) struct DbReferenceHolder {
    inner: Option<DbReference>,
}

impl DbReferenceHolder {
    pub fn new(db: DB) -> Self {
        Self {
            inner: Some(Arc::new(RefCell::new(db))),
        }
    }

    pub fn get(&self) -> Option<&DbReference> {
        self.inner.as_ref()
    }

    pub fn close(&mut self) {
        if let Some(db) = self.inner.take().and_then(Arc::into_inner) {
            db.borrow_mut().cancel_all_background_work(true);
        }
    }
}

impl Drop for DbReferenceHolder {
    fn drop(&mut self) {
        self.close();
    }
}
