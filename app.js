const express = require('express');
const path = require('path')
const app = express();
const port = process.env.PORT | 3000


const publicDirectoryPath = path.join(__dirname, 'dist')

app.use('/dist',express.static(publicDirectoryPath))

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname,'index.html'))
})



app.listen(port,()=>{
    console.log(`Server start on port: ${port}, http://localhost:${port}`)
})
