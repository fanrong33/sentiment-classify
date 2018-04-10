<?php

    $output = '';
    ini_set('display_errors', 'off');
    if($_POST){

        if($_FILES['file']['error'] == 0){
            $pieces = explode('/', $_FILES['file']["type"]);
            $extension = $pieces[1];
            $time = time();

            $file = "upload/".$time.'.'.$extension;
            move_uploaded_file($_FILES["file"]["tmp_name"],  $file);
        }

        if($_POST['url']){
            $file = $_POST['url'];
        }

        $output = shell_exec("sudo -tt /root/anaconda3/envs/torch/bin/python dogcat-master/predict.py '$file' 2>&1");
        // var_dump($output);
    }
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>猫狗识别</title>
    <meta name="viewport" content="width=device-width, minimum-scale=1.0, maximum-scale=1.0">
    <link href="https://cdn.bootcss.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
      <div class="row" style="margin-top:20px;">
          <div class="col-xs-12 col-md-4">
              <img src="<?php echo $file; ?>" alt="" style="max-width: 100%;">
          </div>
          <div class="col-xs-12 col-md-4">
                <form action="" method="POST" enctype="multipart/form-data">
                  <div class="form-group">
                    <label for="exampleInputEmail1">猫狗识别</label>
                    <input type="text" class="form-control" name="url" id="" value="" placeholder="请输入网络图片URL">
                  </div>
                  <button type="submit" class="btn btn-success">开始</button>
                  <div style="display: inline-block;position: relative;">
                        <input id="file" type="file" name="file" accept="image/png, image/bmp, image/jpg, image/jpeg" style="position: absolute;padding: 7px;opacity: 0;" />
                        <button type="button" class="btn btn-success">本地上传</button>
                  </div>
                </form>
          </div>
          <div class="col-xs-12 col-md-4">
            <p>
                结果: <br/>
                <h4>
                  <?php
                    if($output){
                        if(trim($output) == '[[1]]'){
                            echo '狗';
                        }else{
                            echo '猫';
                        }
                    }
                ?>
                </h4>
            </p>
          </div>
      </div>
    </div>
    <script>
    document.getElementById('file').addEventListener('change', function(){
        document.getElementsByTagName('form')[0].submit();
    });
    </script>
</body>
</html>