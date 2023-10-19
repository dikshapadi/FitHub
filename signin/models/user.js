const mongoose = require('mongoose');
const Schema = mongoose.Schema;

userSchema = new Schema( {
	
	unique_id: Number,
	name: String,
	age: Number,
	sex: String,
	height: Number,
	email: String,
	username: String,
	password: String,
	passwordConf: String,
	createdAt: {
		type: Date,
		default: Date.now
	}
}),
User = mongoose.model('User', userSchema);

module.exports = User;