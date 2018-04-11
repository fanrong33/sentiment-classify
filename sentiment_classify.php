<?php

    $output = '';
    ini_set('display_errors', 'off');
    if($_POST){
        
        $sentence = $_POST['sentence'];
        
        $output = shell_exec("sudo -tt /root/anaconda3/envs/torch/bin/python sentiment-classify-master/predict.py '$sentence' 2>&1");
        // var_dump($output);
    }
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>情感分析.酒店评论</title>
    <meta name="viewport" content="width=device-width, minimum-scale=1.0, maximum-scale=1.0">
    <link href="https://cdn.bootcss.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
      <div class="row">
          <div class="col-xs-12 col-md-4" style="margin-top:20px;">
                <form action="" method="POST" enctype="multipart/form-data">
                  <div class="form-group">
                    <label for="exampleInputEmail1">请输入一段想分析的文本: <a href="">随机示例</a></label>
                    <input type="text" class="form-control" name="sentence" value="<?php echo $_POST['sentence']; ?>" placeholder="">
                  </div>
                  <button type="submit" class="btn btn-success">开始</button>
                </form>
          </div>
          <div class="col-xs-12 col-md-4" style="margin-top:20px;">
            <p>
                分析结果: <br/>
                <h4>
                  <?php
                    var_dump($output);
                ?>
                </h4>
            </p>
          </div>
      </div>
    </div>
</body>
</html>